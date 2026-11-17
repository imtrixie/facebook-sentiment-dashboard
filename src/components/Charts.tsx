import React, { useMemo, useState } from 'react';
import { PredictedPost, Sentiment } from '../types/dashboard';

export const SENTIMENT_META: Record<
  Sentiment,
  { label: string; shortLabel: string; color: string; bgSoft: string; textClass: string }
> = {
  positive: {
    label: 'Positive (Tích cực)',
    shortLabel: 'Tích cực',
    color: '#16A34A',
    bgSoft: 'bg-emerald-50 border-emerald-200 text-emerald-800',
    textClass: 'text-emerald-700',
  },
  neutral: {
    label: 'Neutral (Trung lập)',
    shortLabel: 'Trung lập',
    color: '#F59E0B',
    bgSoft: 'bg-amber-50 border-amber-200 text-amber-800',
    textClass: 'text-amber-700',
  },
  negative: {
    label: 'Negative (Tiêu cực)',
    shortLabel: 'Tiêu cực',
    color: '#DC2626',
    bgSoft: 'bg-red-50 border-red-200 text-red-800',
    textClass: 'text-red-700',
  },
};

const SENTIMENTS: Sentiment[] = ['positive', 'neutral', 'negative'];

interface SentimentDistributionProps {
  posts: PredictedPost[];
}

export const SentimentDistributionCharts: React.FC<SentimentDistributionProps> = ({ posts }) => {
  const [hoveredSentiment, setHoveredSentiment] = useState<Sentiment | null>(null);

  const stats = useMemo(() => {
    const counts: Record<Sentiment, number> = { positive: 0, neutral: 0, negative: 0 };
    for (const p of posts) {
      counts[p.sentiment]++;
    }
    const total = posts.length;
    const maxCount = Math.max(counts.positive, counts.neutral, counts.negative, 1);

    return SENTIMENTS.map((s) => ({
      sentiment: s,
      label: SENTIMENT_META[s].label,
      shortLabel: SENTIMENT_META[s].shortLabel,
      color: SENTIMENT_META[s].color,
      count: counts[s],
      pct: total > 0 ? (counts[s] / total) * 100 : 0,
      heightRatio: counts[s] / maxCount,
    }));
  }, [posts]);

  const totalPosts = posts.length;

  // Tính toán các cung tròn cho Donut Chart
  const donutSlices = useMemo(() => {
    const radius = 72;
    const circumference = 2 * Math.PI * radius;
    let cumulativePct = 0;

    return stats.map((item) => {
      const fraction = totalPosts > 0 ? item.count / totalPosts : 0;
      const dashLength = fraction * circumference;
      const offset = -cumulativePct * circumference;
      cumulativePct += fraction;
      return {
        ...item,
        radius,
        circumference,
        dashLength,
        offset,
      };
    });
  }, [stats, totalPosts]);

  // Giá trị tối đa làm tròn đẹp cho trục Y của Bar Chart
  const maxY = useMemo(() => {
    const rawMax = Math.max(...stats.map((s) => s.count), 10);
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawMax)));
    return Math.ceil((rawMax * 1.15) / (magnitude / 2)) * (magnitude / 2);
  }, [stats]);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((r) => Math.round(maxY * r));

  return (
    <section aria-labelledby="section-distribution" className="space-y-4">
      <div className="border-b border-slate-200 pb-2">
        <h2 id="section-distribution" className="text-base font-semibold text-slate-900">
          B. Biểu đồ phân bố cảm xúc
        </h2>
        <p className="text-sm text-slate-500">
          So sánh số lượng và tỷ trọng ba nhóm cảm xúc (Positive: Xanh lá, Neutral: Cam, Negative: Đỏ).
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Biểu đồ cột (Bar Chart) */}
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-800">
              Số lượng bài đăng theo nhóm cảm xúc
            </h3>
            <span className="text-xs text-slate-500 tabular-nums">
              Tổng: {totalPosts.toLocaleString('vi-VN')} bài
            </span>
          </div>

          <div className="relative h-72 w-full">
            <svg
              viewBox="0 0 460 260"
              className="h-full w-full overflow-visible"
              role="img"
              aria-label="Biểu đồ cột số lượng bài đăng theo nhóm cảm xúc"
            >
              {/* Đường lưới ngang và nhãn trục Y */}
              {yTicks.map((tick) => {
                const yPos = 215 - (tick / maxY) * 175;
                return (
                  <g key={tick}>
                    <line
                      x1="54"
                      x2="440"
                      y1={yPos}
                      y2={yPos}
                      stroke="#E2E8F0"
                      strokeDasharray={tick === 0 ? undefined : '3 3'}
                    />
                    <text
                      x="46"
                      y={yPos + 4}
                      textAnchor="end"
                      className="fill-slate-500 text-[11px] tabular-nums"
                    >
                      {tick.toLocaleString('vi-VN')}
                    </text>
                  </g>
                );
              })}

              {/* Các cột dữ liệu */}
              {stats.map((item, idx) => {
                const barWidth = 68;
                const centerX = 115 + idx * 125;
                const barHeight = maxY > 0 ? (item.count / maxY) * 175 : 0;
                const yTop = 215 - barHeight;
                const isHovered = hoveredSentiment === item.sentiment;

                return (
                  <g
                    key={item.sentiment}
                    className="cursor-pointer transition-opacity"
                    style={{ opacity: hoveredSentiment && !isHovered ? 0.55 : 1 }}
                    onMouseEnter={() => setHoveredSentiment(item.sentiment)}
                    onMouseLeave={() => setHoveredSentiment(null)}
                  >
                    <rect
                      x={centerX - barWidth / 2}
                      y={yTop}
                      width={barWidth}
                      height={Math.max(barHeight, 2)}
                      rx={4}
                      fill={item.color}
                    />
                    {/* Giá trị trên đỉnh cột */}
                    <text
                      x={centerX}
                      y={Math.max(yTop - 8, 16)}
                      textAnchor="middle"
                      className="fill-slate-900 text-[12px] font-semibold tabular-nums"
                    >
                      {item.count.toLocaleString('vi-VN')}
                    </text>
                    {/* Nhãn trục X */}
                    <text
                      x={centerX}
                      y={236}
                      textAnchor="middle"
                      className="fill-slate-700 text-[12px] font-medium"
                    >
                      {item.shortLabel}
                    </text>
                    <text
                      x={centerX}
                      y={251}
                      textAnchor="middle"
                      className="fill-slate-400 text-[11px] tabular-nums"
                    >
                      ({item.pct.toFixed(1)}%)
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Biểu đồ tròn (Donut Chart) */}
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-800">
              Tỷ trọng phần trăm các nhóm cảm xúc
            </h3>
            {hoveredSentiment && (
              <span className="text-xs font-medium text-slate-700 tabular-nums">
                {SENTIMENT_META[hoveredSentiment].shortLabel}:{' '}
                {stats
                  .find((s) => s.sentiment === hoveredSentiment)
                  ?.count.toLocaleString('vi-VN')}{' '}
                ({stats.find((s) => s.sentiment === hoveredSentiment)?.pct.toFixed(1)}%)
              </span>
            )}
          </div>

          <div className="flex flex-col items-center justify-center gap-6 sm:flex-row">
            <div className="relative h-56 w-56 shrink-0">
              <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
                <circle
                  cx="100"
                  cy="100"
                  r="72"
                  fill="none"
                  stroke="#F1F5F9"
                  strokeWidth="28"
                />
                {donutSlices.map((slice) => {
                  const isHovered = hoveredSentiment === slice.sentiment;
                  return (
                    <circle
                      key={slice.sentiment}
                      cx="100"
                      cy="100"
                      r={slice.radius}
                      fill="none"
                      stroke={slice.color}
                      strokeWidth={isHovered ? 32 : 28}
                      strokeDasharray={`${slice.dashLength} ${slice.circumference}`}
                      strokeDashoffset={slice.offset}
                      className="cursor-pointer transition-all duration-150"
                      style={{
                        opacity: hoveredSentiment && !isHovered ? 0.55 : 1,
                      }}
                      onMouseEnter={() => setHoveredSentiment(slice.sentiment)}
                      onMouseLeave={() => setHoveredSentiment(null)}
                    />
                  );
                })}
              </svg>
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
                <span className="text-xs font-medium text-slate-500">Tổng bài</span>
                <span className="text-xl font-bold text-slate-900 tabular-nums">
                  {totalPosts.toLocaleString('vi-VN')}
                </span>
              </div>
            </div>

            <div className="w-full max-w-xs space-y-2.5">
              {stats.map((item) => {
                const isHovered = hoveredSentiment === item.sentiment;
                return (
                  <div
                    key={item.sentiment}
                    onMouseEnter={() => setHoveredSentiment(item.sentiment)}
                    onMouseLeave={() => setHoveredSentiment(null)}
                    className={`flex cursor-pointer items-center justify-between rounded-md border px-3 py-2 transition-colors ${
                      isHovered
                        ? 'border-slate-300 bg-slate-50'
                        : 'border-slate-200 bg-white'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span
                        className="h-3 w-3 shrink-0 rounded-sm"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm font-medium text-slate-800">
                        {item.label}
                      </span>
                    </div>
                    <div className="text-right tabular-nums">
                      <span className="text-sm font-semibold text-slate-900">
                        {item.pct.toFixed(1)}%
                      </span>
                      <span className="ml-1.5 text-xs text-slate-500">
                        ({item.count.toLocaleString('vi-VN')})
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

// ============================================================================
// C. PHÂN TÍCH MỨC ĐỘ TƯƠNG TÁC THEO CẢM XÚC
// ============================================================================
type EngagementMode = 'average' | 'total';

const METRICS_CONFIG = [
  { key: 'likes' as const, label: 'Likes', color: '#2563EB' },
  { key: 'comments' as const, label: 'Comments', color: '#0EA5E9' },
  { key: 'shares' as const, label: 'Shares', color: '#64748B' },
];

export const EngagementAnalysisSection: React.FC<{ posts: PredictedPost[] }> = ({ posts }) => {
  const [mode, setMode] = useState<EngagementMode>('average');
  const [hoverKey, setHoverKey] = useState<string | null>(null);

  const summary = useMemo(() => {
    const map: Record<
      Sentiment,
      {
        count: number;
        likesSum: number;
        commentsSum: number;
        sharesSum: number;
        likesAvg: number;
        commentsAvg: number;
        sharesAvg: number;
      }
    > = {
      positive: { count: 0, likesSum: 0, commentsSum: 0, sharesSum: 0, likesAvg: 0, commentsAvg: 0, sharesAvg: 0 },
      neutral: { count: 0, likesSum: 0, commentsSum: 0, sharesSum: 0, likesAvg: 0, commentsAvg: 0, sharesAvg: 0 },
      negative: { count: 0, likesSum: 0, commentsSum: 0, sharesSum: 0, likesAvg: 0, commentsAvg: 0, sharesAvg: 0 },
    };

    for (const p of posts) {
      const bucket = map[p.sentiment];
      bucket.count++;
      bucket.likesSum += p.likes;
      bucket.commentsSum += p.comments;
      bucket.sharesSum += p.shares;
    }

    for (const s of SENTIMENTS) {
      const b = map[s];
      if (b.count > 0) {
        b.likesAvg = b.likesSum / b.count;
        b.commentsAvg = b.commentsSum / b.count;
        b.sharesAvg = b.sharesSum / b.count;
      }
    }
    return map;
  }, [posts]);

  const maxChartVal = useMemo(() => {
    let maxVal = 0;
    for (const s of SENTIMENTS) {
      const b = summary[s];
      const vals =
        mode === 'average'
          ? [b.likesAvg, b.commentsAvg, b.sharesAvg]
          : [b.likesSum, b.commentsSum, b.sharesSum];
      maxVal = Math.max(maxVal, ...vals);
    }
    if (maxVal <= 0) return 10;
    if (mode === 'average') {
      return Math.ceil((maxVal * 1.18) / 5) * 5;
    }
    const magnitude = Math.pow(10, Math.floor(Math.log10(maxVal)));
    return Math.ceil((maxVal * 1.15) / (magnitude / 2)) * (magnitude / 2);
  }, [summary, mode]);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((r) => maxChartVal * r);

  const formatMetric = (val: number) =>
    mode === 'average'
      ? val.toLocaleString('vi-VN', { minimumFractionDigits: 1, maximumFractionDigits: 1 })
      : Math.round(val).toLocaleString('vi-VN');

  return (
    <section aria-labelledby="section-engagement" className="space-y-4">
      <div className="flex flex-col justify-between gap-3 border-b border-slate-200 pb-3 sm:flex-row sm:items-end">
        <div>
          <h2 id="section-engagement" className="text-base font-semibold text-slate-900">
            C. Phân tích mức độ tương tác theo cảm xúc
          </h2>
          <p className="text-sm text-slate-500">
            So sánh Likes, Comments và Shares giữa ba nhóm cảm xúc qua biểu đồ cột nhóm và bảng tổng hợp.
          </p>
        </div>

        <div
          role="group"
          aria-label="Chế độ tính tương tác"
          className="inline-flex rounded-md border border-slate-200 bg-slate-100 p-0.5 self-start sm:self-auto"
        >
          <button
            type="button"
            onClick={() => setMode('average')}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap ${
              mode === 'average'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Trung bình / bài đăng
          </button>
          <button
            type="button"
            onClick={() => setMode('total')}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap ${
              mode === 'total'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Tổng lượng tương tác
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* Biểu đồ cột nhóm */}
        <div className="rounded-lg border border-slate-200 bg-white p-5 xl:col-span-7">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-800">
              {mode === 'average'
                ? 'Trung bình tương tác trên mỗi bài đăng'
                : 'Tổng lượng tương tác theo nhóm cảm xúc'}
            </h3>
            <div className="flex items-center gap-4 text-xs text-slate-600">
              {METRICS_CONFIG.map((m) => (
                <span key={m.key} className="flex items-center gap-1.5">
                  <span
                    className="h-2.5 w-2.5 rounded-xs"
                    style={{ backgroundColor: m.color }}
                  />
                  {m.label}
                </span>
              ))}
            </div>
          </div>

          <div className="h-72 w-full">
            <svg viewBox="0 0 540 260" className="h-full w-full overflow-visible">
              {yTicks.map((tick, i) => {
                const yPos = 215 - (tick / maxChartVal) * 175;
                return (
                  <g key={i}>
                    <line
                      x1="60"
                      x2="520"
                      y1={yPos}
                      y2={yPos}
                      stroke="#E2E8F0"
                      strokeDasharray={i === 0 ? undefined : '3 3'}
                    />
                    <text
                      x="52"
                      y={yPos + 4}
                      textAnchor="end"
                      className="fill-slate-500 text-[11px] tabular-nums"
                    >
                      {mode === 'average'
                        ? tick.toFixed(0)
                        : Math.round(tick).toLocaleString('vi-VN')}
                    </text>
                  </g>
                );
              })}

              {SENTIMENTS.map((sentiment, sIdx) => {
                const groupCenter = 140 + sIdx * 150;
                const b = summary[sentiment];
                const values = {
                  likes: mode === 'average' ? b.likesAvg : b.likesSum,
                  comments: mode === 'average' ? b.commentsAvg : b.commentsSum,
                  shares: mode === 'average' ? b.sharesAvg : b.sharesSum,
                };

                return (
                  <g key={sentiment}>
                    {METRICS_CONFIG.map((metric, mIdx) => {
                      const val = values[metric.key];
                      const barWidth = 30;
                      const gap = 6;
                      const x = groupCenter + (mIdx - 1) * (barWidth + gap) - barWidth / 2;
                      const barHeight = maxChartVal > 0 ? (val / maxChartVal) * 175 : 0;
                      const yTop = 215 - barHeight;
                      const id = `${sentiment}-${metric.key}`;
                      const isHovered = hoverKey === id;

                      return (
                        <g
                          key={metric.key}
                          className="cursor-pointer"
                          onMouseEnter={() => setHoverKey(id)}
                          onMouseLeave={() => setHoverKey(null)}
                        >
                          <rect
                            x={x}
                            y={yTop}
                            width={barWidth}
                            height={Math.max(barHeight, 2)}
                            rx={3}
                            fill={metric.color}
                            style={{ opacity: hoverKey && !isHovered ? 0.6 : 1 }}
                          />
                          <text
                            x={x + barWidth / 2}
                            y={Math.max(yTop - 6, 14)}
                            textAnchor="middle"
                            className={`text-[10px] font-semibold tabular-nums ${
                              isHovered ? 'fill-slate-900' : 'fill-slate-600'
                            }`}
                          >
                            {formatMetric(val)}
                          </text>
                        </g>
                      );
                    })}

                    <text
                      x={groupCenter}
                      y={238}
                      textAnchor="middle"
                      className="fill-slate-800 text-[12px] font-semibold"
                    >
                      {SENTIMENT_META[sentiment].shortLabel}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Bảng thống kê tổng hợp */}
        <div className="rounded-lg border border-slate-200 bg-white p-5 xl:col-span-5">
          <h3 className="mb-3 text-sm font-semibold text-slate-800">
            Bảng thống kê tổng hợp chỉ số tương tác
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-2 font-semibold">Cảm xúc</th>
                  <th className="px-2 py-2 text-right font-semibold">Số bài</th>
                  <th className="px-2 py-2 text-right font-semibold">Likes (TB / Tổng)</th>
                  <th className="px-2 py-2 text-right font-semibold">Comments (TB / Tổng)</th>
                  <th className="pl-2 py-2 text-right font-semibold">Shares (TB / Tổng)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 tabular-nums">
                {SENTIMENTS.map((s) => {
                  const row = summary[s];
                  return (
                    <tr key={s} className="hover:bg-slate-50">
                      <td className="py-3 pr-2 font-medium text-slate-900">
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="h-2.5 w-2.5 rounded-xs"
                            style={{ backgroundColor: SENTIMENT_META[s].color }}
                          />
                          {SENTIMENT_META[s].shortLabel}
                        </span>
                      </td>
                      <td className="px-2 py-3 text-right font-medium text-slate-700">
                        {row.count.toLocaleString('vi-VN')}
                      </td>
                      <td className="px-2 py-3 text-right">
                        <div className="font-semibold text-slate-900">
                          {row.likesAvg.toFixed(1)}
                        </div>
                        <div className="text-[11px] text-slate-500">
                          {row.likesSum.toLocaleString('vi-VN')}
                        </div>
                      </td>
                      <td className="px-2 py-3 text-right">
                        <div className="font-semibold text-slate-900">
                          {row.commentsAvg.toFixed(1)}
                        </div>
                        <div className="text-[11px] text-slate-500">
                          {row.commentsSum.toLocaleString('vi-VN')}
                        </div>
                      </td>
                      <td className="pl-2 py-3 text-right">
                        <div className="font-semibold text-slate-900">
                          {row.sharesAvg.toFixed(1)}
                        </div>
                        <div className="text-[11px] text-slate-500">
                          {row.sharesSum.toLocaleString('vi-VN')}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Dòng trên biểu thị giá trị trung bình mỗi bài đăng; dòng dưới biểu thị tổng lượt tương tác tích lũy.
          </p>
        </div>
      </div>
    </section>
  );
};
