function pearson(pairs) {
  const n = pairs.length;
  if (n < 4) return null;
  let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
  for (const [x, y] of pairs) {
    sx += x; sy += y; sxy += x * y; sx2 += x * x; sy2 += y * y;
  }
  const denom = Math.sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy));
  return denom === 0 ? null : (n * sxy - sx * sy) / denom;
}

function rollingFixedLagCorr(a, b, lag, window) {
  const n = a.length, half = window >> 1;
  return a.map((_, i) => {
    const pairs = [];
    for (let j = i - half + (window % 2 === 0); j <= i + half; j++) {
      const k = j - lag;
      if (j >= 0 && j < n && k >= 0 && k < n && a[j] != null && b[k] != null)
        pairs.push([a[j], b[k]]);
    }
    return pearson(pairs);
  });
}
