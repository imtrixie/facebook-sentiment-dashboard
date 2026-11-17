import { FacebookPost, Sentiment } from '../types/dashboard';

const VALID_SENTIMENTS = new Set<string>(['positive', 'neutral', 'negative']);

export function parseCsvRows(csvText: string): string[][] {
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentCell = '';
  let inQuotes = false;

  const text = csvText.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];

    if (inQuotes) {
      if (ch === '"' && next === '"') {
        currentCell += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        currentCell += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ',') {
        currentRow.push(currentCell);
        currentCell = '';
      } else if (ch === '\r' && next === '\n') {
        currentRow.push(currentCell);
        rows.push(currentRow);
        currentRow = [];
        currentCell = '';
        i++;
      } else if (ch === '\n' || ch === '\r') {
        currentRow.push(currentCell);
        rows.push(currentRow);
        currentRow = [];
        currentCell = '';
      } else {
        currentCell += ch;
      }
    }
  }

  if (currentCell.length > 0 || currentRow.length > 0) {
    currentRow.push(currentCell);
    rows.push(currentRow);
  }

  return rows.filter((row) => row.some((cell) => cell.trim() !== ''));
}

export function parseFacebookDataset(csvText: string): FacebookPost[] {
  const rows = parseCsvRows(csvText);
  if (rows.length < 2) {
    throw new Error('File CSV trống hoặc không có dòng dữ liệu hợp lệ.');
  }

  const header = rows[0].map((col) => col.trim().toLowerCase());
  const requiredCols = [
    'post_id',
    'timestamp',
    'text',
    'category',
    'sentiment',
    'language',
    'has_link',
    'likes',
    'comments',
    'shares',
    'is_spam',
  ];

  const missingCols = requiredCols.filter((col) => !header.includes(col));
  if (missingCols.length > 0) {
    throw new Error(`File CSV thiếu các cột bắt buộc: ${missingCols.join(', ')}`);
  }

  const idx = Object.fromEntries(header.map((h, i) => [h, i])) as Record<string, number>;
  const posts: FacebookPost[] = [];

  for (let r = 1; r < rows.length; r++) {
    const row = rows[r];
    const rawSentiment = (row[idx.sentiment] ?? '').trim().toLowerCase();
    if (!VALID_SENTIMENTS.has(rawSentiment)) {
      continue;
    }

    const rawTimestamp = (row[idx.timestamp] ?? '').trim();
    const parsedTime = Date.parse(rawTimestamp.replace(' ', 'T'));

    const toSafeInt = (val: string | undefined): number => {
      const parsed = Number.parseInt((val ?? '0').trim(), 10);
      return Number.isFinite(parsed) ? parsed : 0;
    };

    posts.push({
      post_id: (row[idx.post_id] ?? String(r)).trim(),
      timestamp: rawTimestamp || 'N/A',
      timestampMs: Number.isFinite(parsedTime) ? parsedTime : 0,
      text: (row[idx.text] ?? '').trim(),
      category: (row[idx.category] ?? 'unknown').trim() || 'unknown',
      sentiment: rawSentiment as Sentiment,
      language: (row[idx.language] ?? 'unknown').trim() || 'unknown',
      has_link: toSafeInt(row[idx.has_link]),
      likes: toSafeInt(row[idx.likes]),
      comments: toSafeInt(row[idx.comments]),
      shares: toSafeInt(row[idx.shares]),
      is_spam: toSafeInt(row[idx.is_spam]),
    });
  }

  if (posts.length === 0) {
    throw new Error('Không có bản ghi nào có nhãn sentiment hợp lệ (positive/neutral/negative).');
  }

  return posts;
}

export function exportRowsToCsv(
  headers: string[],
  rows: (string | number)[][],
  filename: string
): void {
  const escapeCell = (value: string | number): string => {
    const str = String(value ?? '');
    if (/[",\r\n]/.test(str)) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const lines = [
    headers.map(escapeCell).join(','),
    ...rows.map((row) => row.map(escapeCell).join(',')),
  ];

  const blob = new Blob(['\uFEFF' + lines.join('\r\n')], {
    type: 'text/csv;charset=utf-8;',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
