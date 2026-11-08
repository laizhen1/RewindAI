// Ignore a completed search if the user has since changed the active lecture.
export function createRequestGuard() {
  let generation = 0;
  return {
    start: () => ++generation,
    invalidate: () => { generation += 1; },
    current: ticket => ticket === generation,
  };
}
