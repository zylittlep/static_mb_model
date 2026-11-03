// Renders a JSON list of [tex, display] pairs (stdin) to a JSON list of HTML strings (stdout) with KaTeX 0.16.11.
const katex = require('./katex.min.js');
let buf = '';
process.stdin.on('data', d => buf += d);
process.stdin.on('end', () => {
  const items = JSON.parse(buf);
  const out = items.map(([tex, display]) => katex.renderToString(tex, {displayMode: display, throwOnError: true, strict: 'ignore'}));
  process.stdout.write(JSON.stringify(out));
});
