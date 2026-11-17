import React, { useMemo, useState } from 'react';
import { ArrowDownUp, CheckCircle2, Download, Search, XCircle } from 'lucide-react';
import { ModelType, PredictedPost } from '../types/dashboard';
import { exportRowsToCsv } from '../utils/csvParser';
import { SENTIMENT_META } from './Charts';

interface PostsTableProps {
  posts: PredictedPost[];
  selectedModel: ModelType;
}

const PAGE_SIZE = 20;

export const PostsTable: React.FC<PostsTableProps> = ({ posts, selectedModel }) => {
  const [keyword, setKeyword] = useState('');
  const [onlyWrong, setOnlyWrong] = useState(false);
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');
  const [page, setPage] = useState(1);

  const processedPosts = useMemo(() => {
    const kw = keyword.trim().toLowerCase();
    const filtered = posts.filter((p) => {
      if (onlyWrong && p.is_correct) return false;
      if (kw) {
        return (
          p.text.toLowerCase().includes(kw) ||
          p.post_id.toLowerCase().includes(kw) ||
          p.category.toLowerCase().includes(kw)
        );
      }
      return true;
    });

    return filtered.sort((a, b) =>
      sortOrder === 'desc' ? b.timestampMs - a.timestampMs : a.timestampMs - b.timestampMs
    );
  }, [posts, keyword, onlyWrong, sortOrder]);

  const totalPages = Math.max(1, Math.ceil(processedPosts.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pagedPosts = useMemo(
    () => processedPosts.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE),
    [processedPosts, safePage]
  );

  const correctCount = useMemo(
    () => processedPosts.filter((p) => p.is_correct).length,
    [processedPosts]
  );

  const handleDownloadCsv = () => {
    const headers = [
      'ID bài đăng',
      'Thời gian đăng',
      'Danh mục',
      'Nội dung bài đăng',
      'Nhãn thực tế',
      'Nhãn dự đoán',
      'Đánh giá',
      'Likes',
      'Comments',
      'Shares',
    ];
    const rows = processedPosts.map((p) => [
      p.post_id,
      p.timestamp,
      p.category,
      p.text,
      SENTIMENT_META[p.sentiment].shortLabel,
      SENTIMENT_META[p.predicted_sentiment].shortLabel,
      p.is_correct ? 'Đúng' : 'Sai',
      p.likes,
      p.comments,
      p.shares,
    ]);
    exportRowsToCsv(headers, rows, 'facebook_sentiment_filtered.csv');
  };

  return (
    <section aria-labelledby="section-posts-table" className="space-y-4">
      <div className="border-b border-slate-200 pb-2">
        <h2 id="section-posts-table" className="text-base font-semibold text-slate-900">
          E. Danh sách bài đăng và kết quả dự đoán cảm xúc
        </h2>
        <p className="text-sm text-slate-500">
          So sánh trực quan nhãn cảm xúc thực tế và nhãn dự đoán ngoài mẫu từ mô hình{' '}
          <span className="font-semibold text-slate-700">{selectedModel}</span>.
        </p>
      </div>

      {/* Thanh công cụ tìm kiếm, lọc lỗi, sắp xếp và xuất CSV */}
      <div className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-1 flex-col gap-3 sm:flex-row sm:items-center">
          <div className="relative flex-1 max-w-md">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={keyword}
              onChange={(e) => {
                setKeyword(e.target.value);
                setPage(1);
              }}
              placeholder="Tìm theo từ khóa nội dung hoặc ID bài đăng..."
              className="w-full rounded-md border border-slate-300 bg-white py-1.5 pl-9 pr-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none"
            />
          </div>

          <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-slate-700 select-none">
            <input
              type="checkbox"
              checked={onlyWrong}
              onChange={(e) => {
                setOnlyWrong(e.target.checked);
                setPage(1);
              }}
              className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-600"
            />
            <span>Chỉ hiển thị bài dự đoán sai</span>
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-700">
            <ArrowDownUp className="h-3.5 w-3.5 text-slate-500" />
            <select
              aria-label="Sắp xếp theo thời gian"
              value={sortOrder}
              onChange={(e) => {
                setSortOrder(e.target.value as 'desc' | 'asc');
                setPage(1);
              }}
              className="bg-transparent text-sm font-medium text-slate-800 focus:outline-none"
            >
              <option value="desc">Thời gian: Mới nhất trước</option>
              <option value="asc">Thời gian: Cũ nhất trước</option>
            </select>
          </div>

          <button
            type="button"
            onClick={handleDownloadCsv}
            disabled={processedPosts.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 whitespace-nowrap"
          >
            <Download className="h-4 w-4" />
            Tải CSV ({processedPosts.length.toLocaleString('vi-VN')})
          </button>
        </div>
      </div>

      {/* Thống kê nhanh của bảng */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 tabular-nums">
        <div>
          Hiển thị <span className="font-semibold text-slate-900">{processedPosts.length.toLocaleString('vi-VN')}</span> bài đăng
          {' · '}
          Dự đoán đúng:{' '}
          <span className="font-semibold text-emerald-700">
            {correctCount.toLocaleString('vi-VN')}
          </span>{' '}
          (
          {processedPosts.length > 0
            ? ((correctCount / processedPosts.length) * 100).toFixed(1)
            : '0.0'}
          %)
          {' · '}
          Dự đoán sai:{' '}
          <span className="font-semibold text-red-700">
            {(processedPosts.length - correctCount).toLocaleString('vi-VN')}
          </span>
        </div>
        <div>
          Trang {safePage} / {totalPages}
        </div>
      </div>

      {/* Bảng dữ liệu mật độ cao */}
      {processedPosts.length === 0 ? (
        <div className="rounded-lg border border-slate-200 bg-white p-8 text-center">
          <p className="text-sm font-medium text-slate-800">
            Không tìm thấy bài đăng nào khớp với điều kiện tìm kiếm.
          </p>
          <button
            type="button"
            onClick={() => {
              setKeyword('');
              setOnlyWrong(false);
            }}
            className="mt-3 rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
          >
            Xóa bộ lọc tìm kiếm
          </button>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                  <th className="px-3 py-2.5 font-semibold whitespace-nowrap">ID</th>
                  <th className="px-3 py-2.5 font-semibold whitespace-nowrap">Thời gian đăng</th>
                  <th className="px-3 py-2.5 font-semibold whitespace-nowrap">Danh mục</th>
                  <th className="px-3 py-2.5 font-semibold min-w-[260px]">Nội dung bài đăng</th>
                  <th className="px-3 py-2.5 font-semibold whitespace-nowrap">Nhãn thực tế</th>
                  <th className="px-3 py-2.5 font-semibold whitespace-nowrap">Nhãn dự đoán</th>
                  <th className="px-3 py-2.5 font-semibold whitespace-nowrap">Đánh giá</th>
                  <th className="px-3 py-2.5 text-right font-semibold whitespace-nowrap">Likes</th>
                  <th className="px-3 py-2.5 text-right font-semibold whitespace-nowrap">Comments</th>
                  <th className="px-3 py-2.5 text-right font-semibold whitespace-nowrap">Shares</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {pagedPosts.map((post) => {
                  const actualMeta = SENTIMENT_META[post.sentiment];
                  const predMeta = SENTIMENT_META[post.predicted_sentiment];

                  return (
                    <tr key={post.post_id} className="hover:bg-slate-50/80 transition-colors">
                      <td className="px-3 py-2 font-mono text-[11px] text-slate-600 tabular-nums whitespace-nowrap">
                        #{post.post_id}
                      </td>
                      <td className="px-3 py-2 font-mono text-[11px] text-slate-600 tabular-nums whitespace-nowrap">
                        {post.timestamp}
                      </td>
                      <td className="px-3 py-2 text-slate-700 whitespace-nowrap">
                        {post.category}
                      </td>
                      <td className="px-3 py-2 text-slate-800 max-w-md">
                        <div className="line-clamp-2 leading-relaxed">{post.text}</div>
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        <span
                          className={`inline-block rounded border px-2 py-0.5 text-[11px] font-medium ${actualMeta.bgSoft}`}
                        >
                          {actualMeta.shortLabel}
                        </span>
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        <span
                          className={`inline-block rounded border border-dashed px-2 py-0.5 text-[11px] font-medium ${predMeta.bgSoft}`}
                        >
                          {predMeta.shortLabel}
                        </span>
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        {post.is_correct ? (
                          <span className="inline-flex items-center gap-1 font-semibold text-emerald-700">
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            Đúng
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 font-semibold text-red-700">
                            <XCircle className="h-3.5 w-3.5" />
                            Sai
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right font-medium text-slate-800 tabular-nums">
                        {post.likes.toLocaleString('vi-VN')}
                      </td>
                      <td className="px-3 py-2 text-right font-medium text-slate-800 tabular-nums">
                        {post.comments.toLocaleString('vi-VN')}
                      </td>
                      <td className="px-3 py-2 text-right font-medium text-slate-800 tabular-nums">
                        {post.shares.toLocaleString('vi-VN')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Điều hướng phân trang */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between border-t border-slate-200 bg-slate-50 px-4 py-2.5 text-xs">
              <span className="text-slate-600 tabular-nums">
                Dòng {((safePage - 1) * PAGE_SIZE + 1).toLocaleString('vi-VN')} –{' '}
                {Math.min(safePage * PAGE_SIZE, processedPosts.length).toLocaleString('vi-VN')} trong{' '}
                {processedPosts.length.toLocaleString('vi-VN')}
              </span>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  disabled={safePage <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="rounded border border-slate-300 bg-white px-2.5 py-1 font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-40"
                >
                  Trước
                </button>
                <button
                  type="button"
                  disabled={safePage >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  className="rounded border border-slate-300 bg-white px-2.5 py-1 font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-40"
                >
                  Tiếp
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
};
