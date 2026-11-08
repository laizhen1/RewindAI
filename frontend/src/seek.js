// Bind pending timestamps to their video so a delayed metadata event cannot seek another lecture.
export function makePendingSeek(item, video) {
  if (item.start_time == null || (typeof item.start_time === 'string' && !item.start_time.trim())) return null;
  const time = Number(item.start_time);
  if (!video || video.available === false || item.available === false || !Number.isFinite(time) || time < 0) return null;
  return { videoId: video.id, time };
}

export function resolvePendingSeek(pending, videoId, duration) {
  if (!pending || pending.videoId !== videoId) return { status: 'unmatched' };
  if (!Number.isFinite(duration)) return { status: 'waiting' };
  if (pending.time > duration) return { status: 'out_of_range' };
  return { status: 'ready', time: pending.time };
}
