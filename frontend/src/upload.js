// The endpoint is same-origin: lecture files are sent only to the local server.
export function uploadLecture(formData, onProgress, makeRequest = () => new XMLHttpRequest()) {
  return new Promise((resolve, reject) => {
    const request = makeRequest();
    request.open('POST', '/api/library/upload');
    request.upload.addEventListener('progress', event => {
      if (event.lengthComputable && event.total > 0) {
        onProgress(Math.min(100, Math.floor(event.loaded * 100 / event.total)));
      }
    });
    request.addEventListener('load', () => {
      let result;
      try { result = JSON.parse(request.responseText); } catch (_) { result = null; }
      if (request.status >= 200 && request.status < 300 && result && typeof result === 'object') {
        resolve(result);
      } else if (request.status >= 200 && request.status < 300) {
        reject(new Error('The server returned an unexpected response. Check the library before retrying.'));
      } else {
        reject(new Error(result?.error || `Import failed (${request.status}). Check the library before retrying.`));
      }
    });
    const interrupted = () => reject(new Error('The connection was interrupted. The import may have completed; reload and check the library before retrying.'));
    request.addEventListener('error', interrupted);
    request.addEventListener('abort', interrupted);
    request.send(formData);
  });
}
