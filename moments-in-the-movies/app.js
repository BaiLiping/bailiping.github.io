const localVideos = [...document.querySelectorAll("video")];

for (const video of localVideos) {
  video.addEventListener("play", () => {
    for (const other of localVideos) {
      if (other !== video && !other.paused) other.pause();
    }
  });
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) return;
  for (const video of localVideos) {
    if (!video.paused) video.pause();
  }
});
