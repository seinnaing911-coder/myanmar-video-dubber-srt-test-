import os, re, json, time, tempfile, subprocess, shutil, asyncio
from functools import lru_cache
from pathlib import Path

import streamlit as st
import requests
import edge_tts

APP_TITLE = "🎬 Myanmar Free Video Dubber"
MAX_MB = 200
MAX_DURATION_SECONDS = 15 * 60
MAX_SRT_MB = 5
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = "gemini-3.8-flash"
OPENROUTER_MODEL = "openrouter/free"
GROK_MODEL = "grok-4.7"
LOCAL_WHISPER_MODEL = "tiny"
CHUNK_SECONDS = 480

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="centered")


def secret(name, default=""):
    try:
        return str(st.secrets.get(name, default) or default).strip()
    except Exception:
        return os.getenv(name, default).strip()


def run_cmd(args, timeout=900):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-2500:])
    return p.stdout.strip()


def validate_final_video(path, expected_duration):
    if not path.exists() or path.stat().st_size < 10_000:
        raise RuntimeError("Final MP4 ဖိုင်မပြည့်စုံပါ။ Output file size ကို စစ်ဆေးပါ။")
    raw = run_cmd(["ffprobe", "-v", "error", "-show_entries",
                   "stream=codec_type,duration", "-of", "json", str(path)])
    streams = json.loads(raw).get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not video_streams:
        raise RuntimeError("Final MP4 ထဲတွင် Video stream မတွေ့ပါ။")
    if not audio_streams:
        raise RuntimeError("Final MP4 ထဲတွင် မြန်မာအသံ Audio stream မတွေ့ပါ။")
    audio_duration = float(audio_streams[0].get("duration") or 0)
    if audio_duration < 0.25:
        raise RuntimeError("Final MP4 Audio duration မမှန်ပါ။ အသံဖိုင်မထွက်လာနိုင်ပါ။")
    probe = subprocess.run(
        [ffmpeg_path(), "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    volume_text = probe.stdout + probe.stderr
    match = re.search(r"mean_volume:\s*(-?[0-9.]+) dB", volume_text)
    mean_volume = float(match.group(1)) if match else -100.0
    if mean_volume <= -50:
        raise RuntimeError("Final MP4 အသံသည် silent ဖြစ်နေပါသည်။")
    duration_note = f"{audio_duration:.1f}s audio, mean volume {mean_volume:.1f} dB"
    if expected_duration > 0 and audio_duration < min(0.25, expected_duration):
        raise RuntimeError("Final MP4 အသံကြာချိန် မပြည့်စုံပါ။")
    return duration_note


def ffmpeg_path():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg မတွေ့ပါ။ packages.txt ထဲတွင် ffmpeg ထည့်ထားကြောင်း စစ်ပါ။")
    return path


def probe_duration(path):
    value = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return max(0.1, float(value))


def has_audio_stream(path):
    raw = run_cmd(["ffprobe", "-v", "error", "-select_streams", "a:0",
                   "-show_entries", "stream=index", "-of", "json", str(path)])
    return bool(json.loads(raw).get("streams") or [])


def extract_audio(video_path, audio_path):
    # Original audio is extracted only for transcription. It is never copied to the final MP4.
    if not has_audio_stream(video_path):
        raise RuntimeError("ဒီ Video ထဲမှာ မူရင်းအသံ Track မရှိပါ။ စကားပြောသံပါဝင်သော Video ကို တင်ပေးပါ။")
    run_cmd([ffmpeg_path(), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "48k", str(audio_path)], timeout=1800)


def split_audio(audio_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([ffmpeg_path(), "-y", "-i", str(audio_path), "-f", "segment",
             "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
             "-c:a", "libmp3lame", "-b:a", "48k", str(out_dir / "chunk_%03d.mp3")], timeout=1800)
    return sorted(out_dir.glob("chunk_*.mp3"))


def groq_transcribe(audio_path, api_key):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    chunks = split_audio(audio_path, audio_path.parent / "audio_chunks")
    result = []
    for index, chunk in enumerate(chunks):
        with open(chunk, "rb") as file_obj:
            response = requests.post(
                url, headers=headers,
                data={"model": GROQ_STT_MODEL, "response_format": "verbose_json", "timestamp_granularities[]": "segment"},
                files={"file": (chunk.name, file_obj, "audio/mpeg")}, timeout=600)
        if response.status_code >= 400:
            raise RuntimeError(f"Groq STT error {response.status_code}: {response.text[:800]}")
        data = response.json()
        offset = index * CHUNK_SECONDS
        segments = data.get("segments") or []
        if segments:
            for item in segments:
                text = (item.get("text") or "").strip()
                if text:
                    result.append({"start": float(item.get("start", 0)) + offset,
                                   "end": float(item.get("end", item.get("start", 0) + 2)) + offset,
                                   "text": text})
        else:
            # Fallback for providers that return text but no segment timestamps.
            chunk_duration = probe_duration(chunk)
            text = (data.get("text") or "").strip()
            if text:
                result.extend(approximate_segments(text, offset, chunk_duration))
    return normalize_segments(result, probe_duration(audio_path))


def approximate_segments(text, offset, duration):
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if p.strip()]
    if not parts:
        parts = [text.strip()] if text.strip() else []
    if not parts:
        return []
    weights = [max(1, len(p)) for p in parts]
    total = sum(weights)
    cursor = 0.0
    result = []
    for part, weight in zip(parts, weights):
        span = max(1.0, duration * weight / total)
        result.append({"start": offset + cursor, "end": offset + min(duration, cursor + span), "text": part})
        cursor += span
    return result


def local_whisper_transcribe(audio_path):
    from faster_whisper import WhisperModel
    model = WhisperModel(secret("LOCAL_WHISPER_MODEL", LOCAL_WHISPER_MODEL),
                         device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), beam_size=1, vad_filter=True)
    return normalize_segments([{"start": s.start, "end": s.end, "text": s.text.strip()}
                               for s in segments if s.text.strip()], probe_duration(audio_path))


def normalize_segments(segments, duration):
    clean = []
    for item in sorted(segments, key=lambda x: float(x.get("start", 0))):
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if not text:
            continue
        start = max(0.0, min(float(duration), float(item.get("start", 0))))
        end = max(start + 0.25, min(float(duration), float(item.get("end", start + 2))))
        if start >= duration:
            continue
        clean.append({"start": start, "end": end, "text": text})
    return clean


def parse_json_array(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\[.*\]", raw, flags=re.S)
        if not match:
            raise RuntimeError("AI ဘာသာပြန်ရလဒ်ကို JSON အဖြစ် ဖတ်မရပါ။")
        data = json.loads(match.group(0))
    if isinstance(data, dict):
        data = data.get("translations") or data.get("items") or []
    if not isinstance(data, list):
        raise RuntimeError("AI ဘာသာပြန်ရလဒ်ပုံစံ မမှန်ပါ။")
    return data


def contains_burmese(text):
    return bool(re.search(r"[\u1000-\u109F]", str(text or "")))


def call_gemini_api(prompt, api_key, model=GEMINI_MODEL):
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        },
        timeout=240,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:700]}")
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned empty content")
    return text


def call_openrouter_api(prompt, api_key):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://streamlit.io", "X-Title": "Myanmar Auto Video Dubbing"},
        json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2}, timeout=240)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:700]}")
    return response.json()["choices"][0]["message"]["content"]


def call_grok_api(prompt, api_key, model=GROK_MODEL):
    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2}, timeout=240)
    if response.status_code >= 400:
        raise RuntimeError(f"Grok error {response.status_code}: {response.text[:700]}")
    return response.json()["choices"][0]["message"]["content"]


@lru_cache(maxsize=256)
def free_translate_text(text, mymemory_email=""):
    """Translate to Burmese through MyMemory autodetect without Google rate limits."""
    errors = []
    for attempt in range(3):
        try:
            params = {"q": text, "langpair": "autodetect|my-MM"}
            if mymemory_email:
                params["de"] = mymemory_email
            response = requests.get(
                "https://api.mymemory.translated.net/get",
                params=params,
                headers={"User-Agent": "Myanmar-Free-Video-Dubber/1.0"},
                timeout=45,
            )
            data = response.json()
            result = ((data.get("responseData") or {}).get("translatedText") or "").strip()
            if response.ok and data.get("responseStatus") == 200 and contains_burmese(result):
                return result
            errors.append(str(data.get("responseDetails") or result or response.status_code))
        except Exception as exc:
            errors.append(str(exc)[:160])
        if attempt < 2:
            time.sleep(1.0 + attempt)
    raise RuntimeError("MyMemory Burmese translation failed: " + " | ".join(errors[:2]))


def free_translate_segments(segments, mymemory_email="", progress_callback=None):
    translated = []
    errors = []
    chunk_size = 6
    for chunk_start in range(0, len(segments), chunk_size):
        batch = segments[chunk_start:chunk_start + chunk_size]
        source_text = "\n".join(seg["text"] for seg in batch)
        try:
            translated_text = free_translate_text(source_text, mymemory_email)
            lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
            if len(lines) != len(batch) or not all(contains_burmese(line) for line in lines):
                raise RuntimeError("context translation did not preserve all line boundaries")
            for seg, burmese in zip(batch, lines):
                translated.append({**seg, "burmese": burmese})
        except Exception as exc:
            errors.append(f"အပိုင်း {chunk_start + 1}: {str(exc)[:120]}")
        if progress_callback:
            progress_callback(min(1.0, (chunk_start + len(batch)) / len(segments)))
    if errors:
        raise RuntimeError("မြန်မာဘာသာပြန် မအောင်မြင်ပါ။ မူရင်းစာသားကို အသံမပြောင်းပါ။ " + " | ".join(errors[:3]))
    return translated, "Free MyMemory autodetect"


def translate_segments(segments, gemini_key="", openrouter_key="", grok_key="", mymemory_email="",
                       gemini_model=GEMINI_MODEL, grok_model=GROK_MODEL, progress_callback=None):
    translated = []
    batch_size = 18
    provider_order = []
    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
        source_lines = [{"id": i, "text": seg["text"]} for i, seg in enumerate(batch)]
        prompt = (
            "Translate each line into natural, concise spoken Burmese for video dubbing. "
            "Preserve meaning, names, emotion and order. Do not add explanations or events. "
            "Keep each translation short enough to speak within its original time slot. "
            "Return ONLY a JSON array with objects containing id and myanmar. Keep every id.\n" +
            json.dumps(source_lines, ensure_ascii=False))
        raw = None
        errors = []
        if gemini_key:
            try:
                raw = call_gemini_api(prompt, gemini_key, gemini_model)
                provider_order.append("Gemini")
            except Exception as exc:
                errors.append(f"Gemini: {str(exc)[:160]}")
        if raw is None and openrouter_key:
            try:
                raw = call_openrouter_api(prompt, openrouter_key)
                provider_order.append("OpenRouter")
            except Exception as exc:
                errors.append(f"OpenRouter: {str(exc)[:160]}")
        if raw is None and grok_key:
            try:
                raw = call_grok_api(prompt, grok_key, grok_model)
                provider_order.append("Grok")
            except Exception as exc:
                errors.append(f"Grok: {str(exc)[:160]}")
        if raw is None:
            if not gemini_key and not openrouter_key:
                return free_translate_segments(segments, mymemory_email, progress_callback)
            try:
                free_translated, _ = free_translate_segments(segments, mymemory_email, progress_callback)
                return free_translated, " / ".join(dict.fromkeys(provider_order + ["Free MyMemory"]))
            except Exception as free_error:
                errors.append(f"MyMemory: {str(free_error)[:160]}")
                raise RuntimeError("မြန်မာဘာသာပြန် API မရပါ။ " + " | ".join(errors))
        rows = parse_json_array(raw)
        mapping = {}
        for row in rows:
            try:
                mapping[int(row.get("id"))] = str(row.get("myanmar", row.get("translation", ""))).strip()
            except Exception:
                continue
        for i, seg in enumerate(batch):
            burmese = mapping.get(i, "")
            if not burmese:
                raise RuntimeError(f"AI ဘာသာပြန်ရလဒ်တွင် အပိုင်း {i + 1} မပါပါ။")
            if not contains_burmese(burmese):
                raise RuntimeError(f"AI ဘာသာပြန်ရလဒ်သည် မြန်မာစာမဟုတ်ပါ (အပိုင်း {i + 1})။")
            translated.append({**seg, "burmese": burmese})
        if progress_callback:
            progress_callback(min(1.0, (start + len(batch)) / len(segments)))
    provider = " / ".join(dict.fromkeys(provider_order)) or "AI"
    return translated, provider


async def tts_save(text, voice, path):
    await edge_tts.Communicate(text, voice).save(str(path))


def stamp(seconds):
    milliseconds = max(0, int(seconds * 1000))
    hours, rem = divmod(milliseconds, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def parse_srt(raw_bytes, video_duration):
    """Read SRT cue text and preserve each cue's exact start/end timeline."""
    try:
        raw = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = raw_bytes.decode("utf-8", errors="replace")
    cues = []
    timestamp = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
        r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        match = timestamp.search(lines[timing_index])
        if not match:
            raise RuntimeError(f"SRT timeline code မမှန်ပါ: {lines[timing_index]}")
        values = [int(value) for value in match.groups()]
        start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
        end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
        subtitle = " ".join(lines[timing_index + 1:])
        subtitle = re.sub(r"<[^>]*>", "", subtitle)
        subtitle = re.sub(r"\{[^}]*\}", "", subtitle)
        subtitle = re.sub(r"\s+", " ", subtitle).strip()
        if not subtitle or end <= start:
            continue
        if start >= video_duration:
            continue
        cues.append({
            "start": max(0.0, start),
            "end": min(video_duration, end),
            "text": subtitle,
            "burmese": subtitle,
        })

    cues.sort(key=lambda item: item["start"])
    clean = []
    previous_end = 0.0
    for index, cue in enumerate(cues, 1):
        if cue["end"] - cue["start"] < 0.05:
            continue
        if cue["start"] < previous_end - 0.02:
            raise RuntimeError(f"SRT အပိုင်း {index} သည် ယခင်အပိုင်းနှင့် ထပ်နေပါသည်။")
        clean.append(cue)
        previous_end = cue["end"]
    if not clean:
        raise RuntimeError("Video duration အတွင်း အသုံးပြုနိုင်သော SRT စာကြောင်း မတွေ့ပါ။")
    return clean


def write_srt(segments, duration, path):
    blocks = []
    for index, seg in enumerate(segments, 1):
        start = min(duration, max(0, seg["start"]))
        end = min(duration, max(start + 0.25, seg["end"]))
        blocks.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{seg['burmese']}\n")
    path.write_text("\n".join(blocks), encoding="utf-8-sig")


def build_voice_track(segments, voice, workdir, duration, progress_callback=None, require_burmese=True):
    tts_dir = workdir / "tts_segments"
    tts_dir.mkdir(exist_ok=True)
    prepared = []
    for index, seg in enumerate(segments):
        if require_burmese and not contains_burmese(seg.get("burmese", "")):
            raise RuntimeError(f"အပိုင်း {index + 1} တွင် မြန်မာစာမရှိပါ။ Final video မထုတ်ပါ။")
        out = tts_dir / f"voice_{index:04d}.mp3"
        asyncio.run(tts_save(seg["burmese"], voice, out))
        if not out.exists() or out.stat().st_size < 1000:
            raise RuntimeError("မြန်မာ Edge TTS အသံဖိုင် မထွက်လာပါ။ Network သို့မဟုတ် TTS service ကို စစ်ပါ။")
        tts_duration = probe_duration(out)
        requested_start = max(0.0, min(duration, float(seg["start"])))
        # Keep the generated Burmese speech at its natural speed. The matching
        # video window is retimed later instead of speeding up the voice.
        prepared.append((out, requested_start, tts_duration))
        if progress_callback:
            progress_callback((index + 1) / len(segments))
    return prepared


def render_dubbed_video(source, segments, prepared_audio, duration, output_path,
                        export_mode="TikTok 9:16", color_style="Natural", slight_crop=True):
    """Cut each SRT cue window and retime only that video window to its TTS duration."""
    if not segments or len(segments) != len(prepared_audio):
        raise RuntimeError("Video/audio segment count မကိုက်ညီပါ။")

    args = [ffmpeg_path(), "-y", "-i", str(source)]
    for audio_path, _, _ in prepared_audio:
        args += ["-i", str(audio_path)]

    video_labels = []
    audio_labels = []
    filters = []
    cursor = 0.0
    clip_index = 0

    def add_gap(gap_start, gap_end):
        nonlocal clip_index
        if gap_end - gap_start <= 0.05:
            return
        filters.append(
            f"[0:v]trim=start={gap_start:.3f}:end={gap_end:.3f},"
            f"setpts=PTS-STARTPTS[v{clip_index}]"
        )
        video_labels.append(f"[v{clip_index}]")
        filters.append(
            f"anullsrc=r=24000:cl=mono,atrim=duration={gap_end - gap_start:.3f},"
            f"asetpts=PTS-STARTPTS[a{clip_index}]"
        )
        audio_labels.append(f"[a{clip_index}]")
        clip_index += 1

    for index, (seg, prepared) in enumerate(zip(segments, prepared_audio)):
        source_start = max(cursor, max(0.0, min(duration, float(seg["start"]))))
        source_end = max(source_start + 0.05, min(duration, float(seg["end"])))
        add_gap(cursor, source_start)
        source_span = source_end - source_start
        voice_duration = max(0.25, float(prepared[2]))
        speed_ratio = voice_duration / source_span
        filters.append(
            f"[0:v]trim=start={source_start:.3f}:end={source_end:.3f},"
            f"setpts=PTS-STARTPTS,setpts=PTS*{speed_ratio:.6f}[v{clip_index}]"
        )
        video_labels.append(f"[v{clip_index}]")
        audio_input_index = index + 1
        filters.append(
            f"[{audio_input_index}:a]aresample=24000,asetpts=PTS-STARTPTS[a{clip_index}]"
        )
        audio_labels.append(f"[a{clip_index}]")
        clip_index += 1
        cursor = source_end

    add_gap(cursor, duration)
    if not video_labels or not audio_labels:
        raise RuntimeError("SRT timeline မှ render လုပ်နိုင်သော video segment မရှိပါ။")

    filters.append("".join(video_labels) + f"concat=n={len(video_labels)}:v=1:a=0[vout]")
    filters.append("".join(audio_labels) + f"concat=n={len(audio_labels)}:v=0:a=1,"
                   "loudnorm=I=-16:TP=-1.5:LRA=7[aout]")

    if export_mode == "TikTok 9:16 (1080x1920)":
        visual_filters = []
        if slight_crop:
            visual_filters.append("crop=iw*0.97:ih*0.97:(iw-ow)/2:(ih-oh)/2")
        visual_filters.append("scale=1080:1920:force_original_aspect_ratio=increase")
        visual_filters.append("crop=1080:1920:(iw-1080)/2:(ih-1920)/2")
        visual_filters.append("setsar=1")
    else:
        visual_filters = []

    if color_style == "Warm":
        visual_filters.append("colorbalance=rs=.05:gs=.02:bs=-.04")
    elif color_style == "Cool":
        visual_filters.append("colorbalance=rs=-.04:gs=.01:bs=.05")
    elif color_style == "Vivid":
        visual_filters.append("eq=contrast=1.04:saturation=1.08:brightness=0.01")

    video_filter = ",".join(visual_filters) if visual_filters else "null"
    if video_filter != "null":
        filters.append(f"[vout]{video_filter}[vfinal]")
        video_map = "[vfinal]"
    else:
        video_map = "[vout]"
    args += ["-filter_complex", ";".join(filters),
             "-map", video_map, "-map", "[aout]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(output_path)]
    run_cmd(args, timeout=7200)


def main():
    st.title(APP_TITLE)
    st.write("Video တင်ပြီး SRT timeline အတိုင်း စာတစ်ကြောင်းချင်းစီကို အသံထုတ်ကာ video အပိုင်းတွေကို အသံကြာချိန်နဲ့ ကိုက်အောင် အနှေးအမြန်ညှိပေးပါမယ်။")
    st.info("SRT တင်ထားရင် SRT ထဲကစာကိုသာ တိုက်ရိုက်အသံထုတ်ပါမယ်။ SRT မတင်ရင် မူရင်းအသံကို စာသားဖော်ထုတ်ပြီး ယခင်ဘာသာပြန် flow ကို အသုံးပြုပါမယ်။")

    groq_key = secret("GROQ_API_KEY")
    saved_gemini_key = secret("GEMINI_API_KEY")
    gemini_model = secret("GEMINI_MODEL", GEMINI_MODEL)
    openrouter_key = secret("OPENROUTER_API_KEY")
    saved_grok_key = secret("GROK_API_KEY")
    grok_model = secret("GROK_MODEL", GROK_MODEL)
    password = secret("APP_PASSWORD")
    if password:
        entered = st.text_input("ဝင်ရောက်ရန် Password", type="password")
        if entered != password:
            st.info("Password ထည့်ပြီးမှ ဆက်လုပ်နိုင်ပါမယ်။")
            st.stop()

    uploaded = st.file_uploader("Video တင်ပါ (MP4 / MOV / MKV / WEBM)", type=["mp4", "mov", "mkv", "webm"])
    uploaded_srt = st.file_uploader(
        "SRT တင်ပါ (Optional — SRT တင်ရင် SRT ထဲကစာကိုသာ အသံထုတ်ပါမယ်)",
        type=["srt", "txt"],
        help="SRT timecode တစ်ခုချင်းစီအတိုင်း video ကို ခွဲပြီး စာကြောင်းအသံကြာချိန်နဲ့ ကိုက်အောင် video speed ကို ပြောင်းပါမယ်။",
    )
    gemini_input_key = st.text_input(
        "Gemini API Key — Translation only (Optional)",
        value="",
        type="password",
        placeholder="AIza...",
        help="Gemini ကို video စာသားကို မြန်မာလို သဘာဝကျကျ ဘာသာပြန်ရန်သာ အသုံးပြုပါမယ်။ STT၊ TTS နှင့် video rendering အတွက် မသုံးပါ။ ဒီနေရာမှာထည့်သော key ကို ဒီ session အတွင်းသာ အသုံးပြုပါမယ်။",
    ).strip()
    gemini_key = gemini_input_key or saved_gemini_key
    grok_input_key = st.text_input(
        "Grok API Key — Translation fallback (Optional)", value="", type="password",
        placeholder="xai-...",
        help="Gemini နှင့် OpenRouter မရမှသာ Burmese translation အတွက် အသုံးပြုပါမယ်။",
    ).strip()
    grok_key = grok_input_key or saved_grok_key
    mymemory_email = st.text_input(
        "MYMEMORY_EMAIL (Optional)",
        value=secret("MYMEMORY_EMAIL"),
        placeholder="your-email@example.com",
        help="MyMemory translation quota ပိုမိုရရှိရန် email ထည့်နိုင်ပါတယ်။ မထည့်လည်း Free mode အလုပ်လုပ်ပါမယ်။",
    ).strip()
    active_providers = []
    if groq_key:
        active_providers.append("Groq Whisper")
    if gemini_key:
        active_providers.append(f"Gemini ({gemini_model})")
    if openrouter_key:
        active_providers.append("OpenRouter fallback")
    if grok_key:
        active_providers.append(f"Grok ({grok_model}) fallback")
    active_providers.append("Free MyMemory fallback")
    if not active_providers:
        active_providers.append("Free local Whisper → Free translation → Edge TTS")
    st.caption("အသုံးပြုမည့် processing mode: " + " → ".join(active_providers))
    voice_label = st.selectbox("မြန်မာအသံရွေးချယ်ပါ", ["နီလာ (မိန်းကလေး)", "သီဟ (ယောက်ျားလေး)"])
    export_mode = st.selectbox("Final video format", ["TikTok 9:16 (1080x1920)", "Original aspect ratio"])
    color_style = st.selectbox("Color style", ["Natural", "Warm", "Cool", "Vivid"])
    slight_crop = st.checkbox("Slight center crop (3%)", value=True,
                              help="Portrait framing အတွက်သာ အသုံးပြုသည်။ ကိုယ်ပိုင် သို့မဟုတ် လိုင်စင်ရှိသော video များအတွက် သုံးပါ။")
    st.caption("ဒီ export controls တွေဟာ platform formatting နဲ့ creative grading အတွက်သာ ဖြစ်ပြီး မူပိုင်ခွင့်စစ်ဆေးမှုကို ရှောင်ရန် မဟုတ်ပါ။")

    if uploaded and uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"Video သည် {MAX_MB} MB ထက်ကျော်နေပါသည်။")
        st.stop()
    if uploaded_srt and uploaded_srt.size > MAX_SRT_MB * 1024 * 1024:
        st.error(f"SRT သည် {MAX_SRT_MB} MB ထက်ကျော်နေပါသည်။")
        st.stop()

    if st.button("🎙️ မြန်မာအသံနဲ့ Dubbing လုပ်မယ်", type="primary", disabled=uploaded is None):
        if not groq_key and uploaded_srt is None:
            st.info("GROQ_API_KEY မတွေ့ပါ။ အသံမှ စာသားဖော်ထုတ်ရန် Free local Whisper mode ကို အသုံးပြုနေပါတယ်။ Gemini API key သည် စာသားဖော်ထုတ်ခြင်းမဟုတ်ဘဲ နောက်အဆင့် မြန်မာဘာသာပြန်ခြင်းအတွက်သာ အသုံးပြုပါမယ်။ ပထမဆုံး run မှာ Whisper model download လုပ်ရနိုင်ပါတယ်။")

        with tempfile.TemporaryDirectory() as temp_dir:
            work = Path(temp_dir)
            video = work / Path(uploaded.name).name
            video.write_bytes(uploaded.getbuffer())
            srt_bytes = uploaded_srt.getvalue() if uploaded_srt else None
            original_audio = work / "original_for_transcription.mp3"
            output_mp3 = work / "myanmar_voice.mp3"
            output_mp4 = work / "myanmar_dubbed_video.mp4"
            duration = probe_duration(video)
            if duration > MAX_DURATION_SECONDS:
                raise RuntimeError(
                    f"Video ကြာချိန်သည် {MAX_DURATION_SECONDS // 60} မိနစ်ထက် မကျော်ရပါ။ "
                    "Streamlit Cloud resource ကန့်သတ်ချက်ကြောင့် video ကို အပိုင်းတိုများခွဲပြီး ပြန်တင်ပါ။"
                )
            progress = st.progress(0, text="Video ပြင်ဆင်နေသည်... (0%)")
            status = st.empty()

            def set_progress(value, label):
                percent = int(max(0, min(100, value)))
                progress.progress(percent, text=f"{label} ({percent}%)")

            try:
                if srt_bytes is not None:
                    status.info("SRT timeline နဲ့ စာသားကို ဖတ်နေသည် — SRT ထဲကစာကိုသာ အသံထုတ်ပါမယ်...")
                    segments = parse_srt(srt_bytes, duration)
                    translation_provider = "SRT direct text (no translation)"
                    set_progress(35, f"SRT အပိုင်း {len(segments)} ခု ဖတ်ပြီးပါပြီ")
                else:
                    status.info("မူရင်းအသံကို စာသားဖော်ထုတ်ရန် ယာယီထုတ်ယူနေသည်...")
                    extract_audio(video, original_audio)
                    set_progress(8, "အသံဖိုင် ပြင်ဆင်ပြီးပါပြီ")

                    status.info("Groq Whisper သို့မဟုတ် Free local Whisper ဖြင့် မူရင်းစကားပြောသံနှင့် အချိန်မှတ်တမ်း ထုတ်ယူနေသည်...")
                    if groq_key:
                        try:
                            segments = groq_transcribe(original_audio, groq_key)
                        except Exception as groq_error:
                            status.warning(f"Groq မအောင်မြင်ပါ။ Local Whisper ကို စမ်းသုံးနေသည်။ {str(groq_error)[:180]}")
                            segments = local_whisper_transcribe(original_audio)
                    else:
                        segments = local_whisper_transcribe(original_audio)
                    if not segments:
                        raise RuntimeError("မူရင်းစကားပြောသံမှ စာသားမရရှိပါ။ Video ထဲတွင် အသံရှိ/မရှိ စစ်ဆေးပါ။")
                    set_progress(35, f"စကားပြောအပိုင်း {len(segments)} ခု ဖော်ထုတ်ပြီးပါပြီ")

                    status.info("စာသားအပိုင်းတိုင်းကို မြန်မာလို ဘာသာပြန်နေသည်...")
                    segments, translation_provider = translate_segments(
                        segments, gemini_key, openrouter_key, grok_key, mymemory_email,
                        gemini_model, grok_model,
                        lambda fraction: set_progress(35 + fraction * 25, "မြန်မာဘာသာပြန်နေသည်"))
                    set_progress(62, f"မြန်မာဘာသာပြန်ပြီးပါပြီ ({translation_provider})")

                voice = "my-MM-NilarNeural" if voice_label.startswith("နီလာ") else "my-MM-ThihaNeural"
                status.info("SRT timeline အပိုင်းတစ်ခုချင်းစီအတွက် အသံဖန်တီးပြီး video speed နဲ့ ညှိနေသည်...")
                prepared_audio = build_voice_track(
                    segments, voice, work, duration,
                    lambda fraction: set_progress(62 + fraction * 20, "စာကြောင်းအသံ ဖန်တီးနေသည်"),
                    require_burmese=srt_bytes is None)
                set_progress(84, "မြန်မာအသံ ပြင်ဆင်ပြီးပါပြီ")

                status.info("မူရင်းအသံ Track ကို ဖယ်ရှားပြီး မြန်မာအသံတစ်ခုတည်း ထည့်နေသည်...")
                render_dubbed_video(video, segments, prepared_audio, duration, output_mp4,
                                    export_mode=export_mode, color_style=color_style,
                                    slight_crop=slight_crop)
                validation_note = validate_final_video(output_mp4, duration)
                # Export a standalone MP3 from the final MP4 audio, ensuring it matches the final mix.
                run_cmd([ffmpeg_path(), "-y", "-i", str(output_mp4), "-vn", "-c:a", "libmp3lame",
                         "-b:a", "160k", str(output_mp3)], timeout=1200)
                if not output_mp3.exists() or output_mp3.stat().st_size < 1000:
                    raise RuntimeError("MP3 အသံဖိုင် မပြည့်စုံပါ။")
                set_progress(100, "ပြီးပါပြီ")
                status.success(f"Dubbing အောင်မြင်ပါပြီ။ Final audio စစ်ဆေးပြီးပါပြီ ({validation_note})")
                st.subheader("Final Myanmar Dubbed Video")
                st.video(str(output_mp4))
                output_name = "myanmar_dubbed_tiktok_9x16.mp4" if export_mode.startswith("TikTok") else "myanmar_dubbed_video.mp4"
                st.download_button("⬇️ မြန်မာအသံ Video (MP4)", output_mp4.read_bytes(),
                                   file_name=output_name, mime="video/mp4")
                st.download_button("⬇️ မြန်မာအသံ (MP3)", output_mp3.read_bytes(),
                                   file_name="myanmar_voice.mp3", mime="audio/mpeg")
            except MemoryError:
                progress.empty()
                status.error("Memory မလုံလောက်ပါ။ Video size/ကြာချိန်ကို လျှော့ပြီး ပြန်စမ်းပါ။")
                st.caption("Streamlit Cloud resource limit ကြောင့် process ရပ်သွားနိုင်ပါသည်။")
            except subprocess.TimeoutExpired:
                progress.empty()
                status.error("လုပ်ဆောင်ချိန်ကျော်လွန်သွားပါသည်။ Video ကို ပိုတိုအောင် ခွဲပြီး ပြန်စမ်းပါ။")
                st.caption("FFmpeg/Whisper processing သည် Cloud runtime limit ထက် ကြာသွားနိုင်ပါသည်။")
            except Exception as exc:
                progress.empty()
                status.error(f"လုပ်ဆောင်မှု မအောင်မြင်ပါ: {exc}")
                st.caption("API quota၊ Network၊ FFmpeg သို့မဟုတ် Streamlit Cloud resource ကန့်သတ်ချက်ကြောင့် ဖြစ်နိုင်ပါတယ်။")


if __name__ == "__main__":
    main()
