import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  Download,
  FileCode2,
  Filter,
  RotateCcw,
  Upload,
  X,
} from 'lucide-react';
import {
  EngagementAnalysisSection,
  SENTIMENT_META,
  SentimentDistributionCharts,
} from './components/Charts';
import { LivePredictor } from './components/LivePredictor';
import { PostsTable } from './components/PostsTable';
import { StreamlitCodeDrawer } from './components/StreamlitCodeDrawer';
import {
  FacebookPost,
  FilterState,
  ModelType,
  PredictedPost,
  Sentiment,
  VectorizerType,
} from './types/dashboard';
import { parseFacebookDataset } from './utils/csvParser';
import { SENTIMENT_CLASSES, TrainedMlSuite, trainMlSuite } from './utils/mlEngine';

export default function App() {
  const [posts, setPosts] = useState<FacebookPost[]>([]);
  const [loadingData, setLoadingData] = useState(true);
  const [dataError, setDataError] = useState<string | null>(null);
  const [dataSourceName, setDataSourceName] = useState('Facebook_data_txt.csv');

  const [selectedModel, setSelectedModel] = useState<ModelType>('Naive Bayes');
  const [vectorizerType, setVectorizerType] = useState<VectorizerType>('TF-IDF');
  const [mlSuite, setMlSuite] = useState<TrainedMlSuite | null>(null);
  const [trainingMl, setTrainingMl] = useState(false);

  const [filters, setFilters] = useState<FilterState>({
    categories: [],
    sentiments: [...SENTIMENT_CLASSES],
    languages: [],
    excludeSpam: false,
  });

  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [codeDrawerOpen, setCodeDrawerOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Tải file Facebook_data_txt.csv khi khởi động
  useEffect(() => {
    let cancelled = false;
    setLoadingData(true);
    setDataError(null);

    fetch('/Facebook_data_txt.csv')
      .then((res) => {
        if (!res.ok) {
          throw new Error(
            'Không tìm thấy file dữ liệu Facebook_data_txt.csv. Vui lòng kiểm tra file dữ liệu hoặc tải file CSV lên.'
          );
        }
        return res.text();
      })
      .then((csvText) => {
        if (cancelled) return;
        const parsed = parseFacebookDataset(csvText);
        const uniqueCats = Array.from(new Set(parsed.map((p) => p.category))).sort();
        const uniqueLangs = Array.from(new Set(parsed.map((p) => p.language))).sort();

        setPosts(parsed);
        setFilters({
          categories: uniqueCats,
          sentiments: [...SENTIMENT_CLASSES],
          languages: uniqueLangs,
          excludeSpam: false,
        });
        setLoadingData(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setDataError(
          err instanceof Error ? err.message : 'Đã xảy ra lỗi khi đọc file CSV.'
        );
        setLoadingData(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Huấn luyện mô hình ML ngoài luồng vẽ UI khi dữ liệu hoặc phương pháp trích xuất đặc trưng thay đổi
  useEffect(() => {
    if (posts.length === 0) {
      setMlSuite(null);
      return;
    }

    setTrainingMl(true);
    const timer = window.setTimeout(() => {
      try {
        const suite = trainMlSuite(posts, vectorizerType);
        setMlSuite(suite);
      } finally {
        setTrainingMl(false);
      }
    }, 20);

    return () => window.clearTimeout(timer);
  }, [posts, vectorizerType]);

  const allCategories = useMemo(
    () => Array.from(new Set(posts.map((p) => p.category))).sort(),
    [posts]
  );

  const allLanguages = useMemo(
    () => Array.from(new Set(posts.map((p) => p.language))).sort(),
    [posts]
  );

  // Gắn nhãn dự đoán ngoài mẫu (OOF) của mô hình đã chọn vào từng bài đăng
  const postsWithPredictions: PredictedPost[] = useMemo(() => {
    if (!mlSuite) return [];
    const oof = mlSuite.evaluations[selectedModel].oofPredictions;
    return posts.map((post, idx) => {
      const predicted = oof[idx] ?? 'neutral';
      return {
        ...post,
        predicted_sentiment: predicted,
        is_correct: post.sentiment === predicted,
      };
    });
  }, [posts, mlSuite, selectedModel]);

  // Lọc đồng bộ toàn bộ Dashboard theo Sidebar
  const filteredPosts = useMemo(() => {
    const catSet = new Set(filters.categories);
    const sentSet = new Set(filters.sentiments);
    const langSet = new Set(filters.languages);

    return postsWithPredictions.filter((p) => {
      if (!catSet.has(p.category)) return false;
      if (!sentSet.has(p.sentiment)) return false;
      if (!langSet.has(p.language)) return false;
      if (filters.excludeSpam && p.is_spam === 1) return false;
      return true;
    });
  }, [postsWithPredictions, filters]);

  // Tính toán KPI tổng quan
  const kpi = useMemo(() => {
    const total = filteredPosts.length;
    let pos = 0;
    let neu = 0;
    let neg = 0;

    for (const p of filteredPosts) {
      if (p.sentiment === 'positive') pos++;
      else if (p.sentiment === 'neutral') neu++;
      else if (p.sentiment === 'negative') neg++;
    }

    const pct = (n: number) => (total > 0 ? ((n / total) * 100).toFixed(1) : '0.0');

    return {
      total,
      pos,
      neu,
      neg,
      posPct: pct(pos),
      neuPct: pct(neu),
      negPct: pct(neg),
    };
  }, [filteredPosts]);

  const handleResetFilters = () => {
    setFilters({
      categories: [...allCategories],
      sentiments: [...SENTIMENT_CLASSES],
      languages: [...allLanguages],
      excludeSpam: false,
    });
  };

  const handleUploadCsv = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = String(reader.result ?? '');
        const parsed = parseFacebookDataset(text);
        const uniqueCats = Array.from(new Set(parsed.map((p) => p.category))).sort();
        const uniqueLangs = Array.from(new Set(parsed.map((p) => p.language))).sort();
        setPosts(parsed);
        setDataSourceName(file.name);
        setDataError(null);
        setFilters({
          categories: uniqueCats,
          sentiments: [...SENTIMENT_CLASSES],
          languages: uniqueLangs,
          excludeSpam: false,
        });
      } catch (err: unknown) {
        setDataError(
          err instanceof Error ? err.message : 'File CSV không hợp lệ.'
        );
      }
    };
    reader.readAsText(file, 'utf-8');
    e.target.value = '';
  };

  const toggleItem = <T extends string>(list: T[], item: T): T[] =>
    list.includes(item) ? list.filter((x) => x !== item) : [...list, item];

  const renderSidebarContent = () => (
    <div className="flex h-full flex-col justify-between">
      <div className="space-y-5 p-4">
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-slate-700" />
            <span className="text-sm font-semibold text-slate-900">Bộ lọc dữ liệu</span>
          </div>
          <button
            type="button"
            onClick={handleResetFilters}
            className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700"
          >
            <RotateCcw className="h-3 w-3" />
            Mặc định
          </button>
        </div>

        {/* Chọn mô hình & Trích xuất đặc trưng */}
        <div className="space-y-2.5 border-b border-slate-200 pb-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Mô hình dự đoán (ML)
          </div>
          <div>
            <label htmlFor="sidebar-model-select" className="mb-1 block text-xs font-medium text-slate-700">
              Thuật toán phân loại
            </label>
            <select
              id="sidebar-model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as ModelType)}
              className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none"
            >
              <option value="Naive Bayes">Naive Bayes (MultinomialNB)</option>
              <option value="Decision Tree">Decision Tree (CART)</option>
            </select>
          </div>
          <div>
            <div className="mb-1 text-xs font-medium text-slate-700">Trích xuất đặc trưng</div>
            <div className="grid grid-cols-2 gap-1.5 rounded-md border border-slate-200 bg-slate-100 p-0.5">
              {(['TF-IDF', 'Bag of Words'] as VectorizerType[]).map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setVectorizerType(v)}
                  className={`rounded py-1 text-xs font-medium transition-colors ${
                    vectorizerType === v
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Bộ lọc Sentiment */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Nhóm cảm xúc
            </span>
            <button
              type="button"
              onClick={() =>
                setFilters((f) => ({
                  ...f,
                  sentiments:
                    f.sentiments.length === SENTIMENT_CLASSES.length
                      ? []
                      : [...SENTIMENT_CLASSES],
                }))
              }
              className="text-[11px] font-medium text-blue-600 hover:underline"
            >
              {filters.sentiments.length === SENTIMENT_CLASSES.length
                ? 'Bỏ chọn tất cả'
                : 'Chọn tất cả'}
            </button>
          </div>
          <div className="space-y-1.5">
            {SENTIMENT_CLASSES.map((s: Sentiment) => (
              <label
                key={s}
                className="flex cursor-pointer items-center justify-between rounded px-1.5 py-1 text-xs text-slate-700 hover:bg-slate-100"
              >
                <span className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={filters.sentiments.includes(s)}
                    onChange={() =>
                      setFilters((f) => ({
                        ...f,
                        sentiments: toggleItem(f.sentiments, s),
                      }))
                    }
                    className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600"
                  />
                  <span
                    className="h-2.5 w-2.5 rounded-xs"
                    style={{ backgroundColor: SENTIMENT_META[s].color }}
                  />
                  <span className="font-medium">{SENTIMENT_META[s].label}</span>
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Bộ lọc Danh mục (Category) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Danh mục ({filters.categories.length}/{allCategories.length})
            </span>
            <button
              type="button"
              onClick={() =>
                setFilters((f) => ({
                  ...f,
                  categories:
                    f.categories.length === allCategories.length ? [] : [...allCategories],
                }))
              }
              className="text-[11px] font-medium text-blue-600 hover:underline"
            >
              {filters.categories.length === allCategories.length
                ? 'Bỏ chọn tất cả'
                : 'Chọn tất cả'}
            </button>
          </div>
          <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
            {allCategories.map((cat) => (
              <label
                key={cat}
                className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-xs text-slate-700 hover:bg-slate-100"
              >
                <input
                  type="checkbox"
                  checked={filters.categories.includes(cat)}
                  onChange={() =>
                    setFilters((f) => ({
                      ...f,
                      categories: toggleItem(f.categories, cat),
                    }))
                  }
                  className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600"
                />
                <span className="truncate">{cat}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Bộ lọc Ngôn ngữ (Language) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Ngôn ngữ ({filters.languages.length}/{allLanguages.length})
            </span>
            <button
              type="button"
              onClick={() =>
                setFilters((f) => ({
                  ...f,
                  languages:
                    f.languages.length === allLanguages.length ? [] : [...allLanguages],
                }))
              }
              className="text-[11px] font-medium text-blue-600 hover:underline"
            >
              {filters.languages.length === allLanguages.length
                ? 'Bỏ chọn tất cả'
                : 'Chọn tất cả'}
            </button>
          </div>
          <div className="grid grid-cols-2 gap-1">
            {allLanguages.map((lang) => (
              <label
                key={lang}
                className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-xs text-slate-700 hover:bg-slate-100"
              >
                <input
                  type="checkbox"
                  checked={filters.languages.includes(lang)}
                  onChange={() =>
                    setFilters((f) => ({
                      ...f,
                      languages: toggleItem(f.languages, lang),
                    }))
                  }
                  className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600"
                />
                <span className="uppercase font-mono">{lang}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Tùy chọn loại bỏ spam */}
        <div className="border-t border-slate-200 pt-3">
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-800">
            <input
              type="checkbox"
              checked={filters.excludeSpam}
              onChange={(e) =>
                setFilters((f) => ({ ...f, excludeSpam: e.target.checked }))
              }
              className="h-4 w-4 rounded border-slate-300 text-blue-600"
            />
            <span>Loại bỏ bài đăng spam (is_spam = 1)</span>
          </label>
        </div>
      </div>

      <div className="border-t border-slate-200 bg-slate-50 p-4 text-xs text-slate-600 tabular-nums">
        Đang hiển thị{' '}
        <span className="font-semibold text-slate-900">
          {filteredPosts.length.toLocaleString('vi-VN')}
        </span>{' '}
        / {posts.length.toLocaleString('vi-VN')} bài đăng
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Top Bar */}
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileSidebarOpen(true)}
              aria-label="Mở bộ lọc"
              className="inline-flex items-center justify-center rounded-md border border-slate-300 p-1.5 text-slate-700 hover:bg-slate-50 lg:hidden"
            >
              <Filter className="h-4 w-4" />
            </button>
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-slate-900 text-white">
              <BarChart3 className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-sm font-bold text-slate-900 sm:text-base">
                Facebook Sentiment Analysis Dashboard
              </h1>
              <p className="truncate text-xs text-slate-500">
                Nguồn dữ liệu: <span className="font-mono text-slate-700">{dataSourceName}</span>
                {' · '}
                <span className="tabular-nums">{posts.length.toLocaleString('vi-VN')}</span> bản ghi
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleUploadCsv}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="hidden md:inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap"
            >
              <Upload className="h-3.5 w-3.5" />
              Nạp CSV khác
            </button>
            <button
              type="button"
              onClick={() => setCodeDrawerOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 hover:bg-slate-50 whitespace-nowrap"
            >
              <FileCode2 className="h-3.5 w-3.5" />
              Mã nguồn Streamlit
            </button>
            <a
              href="/streamlit/streamlit-facebook-sentiment.zip"
              download="streamlit-facebook-sentiment.zip"
              className="inline-flex items-center gap-1.5 rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800 whitespace-nowrap"
            >
              <Download className="h-3.5 w-3.5" />
              Tải Streamlit (.zip)
            </a>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1600px]">
        {/* Sidebar Desktop */}
        <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-white lg:block">
          <div className="sticky top-[57px] h-[calc(100vh-57px)] overflow-y-auto">
            {renderSidebarContent()}
          </div>
        </aside>

        {/* Sidebar Mobile Drawer */}
        {mobileSidebarOpen && (
          <div className="fixed inset-0 z-40 flex lg:hidden">
            <div
              className="fixed inset-0 bg-slate-900/40"
              onClick={() => setMobileSidebarOpen(false)}
            />
            <aside className="relative z-10 h-full w-72 overflow-y-auto bg-white shadow-xl">
              <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
                <span className="text-sm font-semibold text-slate-900">Bộ lọc</span>
                <button
                  type="button"
                  onClick={() => setMobileSidebarOpen(false)}
                  className="rounded p-1 text-slate-500 hover:bg-slate-100"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              {renderSidebarContent()}
            </aside>
          </div>
        )}

        {/* Nội dung chính Dashboard */}
        <main className="min-w-0 flex-1 space-y-8 p-4 sm:p-6 lg:p-8">
          {dataError ? (
            <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-red-900">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
                <div>
                  <h2 className="text-sm font-semibold">Không thể tải dữ liệu CSV</h2>
                  <p className="mt-1 text-sm text-red-800">{dataError}</p>
                </div>
              </div>
            </div>
          ) : loadingData || trainingMl || !mlSuite ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {[1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="h-24 animate-pulse rounded-lg border border-slate-200 bg-white p-4"
                  />
                ))}
              </div>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div className="h-80 animate-pulse rounded-lg border border-slate-200 bg-white" />
                <div className="h-80 animate-pulse rounded-lg border border-slate-200 bg-white" />
              </div>
            </div>
          ) : filteredPosts.length === 0 ? (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-8 text-center">
              <AlertTriangle className="mx-auto h-8 w-8 text-amber-600" />
              <h2 className="mt-3 text-base font-semibold text-slate-900">
                Không có dữ liệu phù hợp với điều kiện lọc
              </h2>
              <p className="mt-1 text-sm text-slate-600">
                Vui lòng chọn thêm danh mục, nhóm cảm xúc, ngôn ngữ hoặc bỏ tùy chọn loại bỏ spam ở thanh bên.
              </p>
              <button
                type="button"
                onClick={handleResetFilters}
                className="mt-4 inline-flex items-center gap-1.5 rounded-md bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-slate-800"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Khôi phục bộ lọc mặc định
              </button>
            </div>
          ) : (
            <>
              {/* A. Thống kê tổng quan (KPI Cards) */}
              <section aria-labelledby="section-kpi" className="space-y-3">
                <div className="border-b border-slate-200 pb-2">
                  <h2 id="section-kpi" className="text-base font-semibold text-slate-900">
                    A. Thống kê tổng quan
                  </h2>
                  <p className="text-sm text-slate-500">
                    Chỉ số KPI cập nhật đồng bộ theo bộ lọc hiện tại.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-lg border border-slate-200 bg-white p-4">
                    <div className="text-xs font-semibold text-slate-500">
                      Tổng số bài đăng
                    </div>
                    <div className="mt-1.5 text-2xl font-bold text-slate-900 tabular-nums">
                      {kpi.total.toLocaleString('vi-VN')}
                    </div>
                    <div className="mt-1 text-xs text-slate-500 tabular-nums">
                      100.0% dữ liệu đang lọc
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
                      <span className="h-2 w-2 rounded-xs bg-emerald-600" />
                      Tích cực (Positive)
                    </div>
                    <div className="mt-1.5 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-bold text-slate-900">
                        {kpi.pos.toLocaleString('vi-VN')}
                      </span>
                      <span className="text-sm font-semibold text-emerald-700">
                        ({kpi.posPct}%)
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      Chiếm {kpi.posPct}% tổng bài đăng
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
                      <span className="h-2 w-2 rounded-xs bg-amber-500" />
                      Trung lập (Neutral)
                    </div>
                    <div className="mt-1.5 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-bold text-slate-900">
                        {kpi.neu.toLocaleString('vi-VN')}
                      </span>
                      <span className="text-sm font-semibold text-amber-700">
                        ({kpi.neuPct}%)
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      Chiếm {kpi.neuPct}% tổng bài đăng
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
                      <span className="h-2 w-2 rounded-xs bg-red-600" />
                      Tiêu cực (Negative)
                    </div>
                    <div className="mt-1.5 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-bold text-slate-900">
                        {kpi.neg.toLocaleString('vi-VN')}
                      </span>
                      <span className="text-sm font-semibold text-red-700">
                        ({kpi.negPct}%)
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      Chiếm {kpi.negPct}% tổng bài đăng
                    </div>
                  </div>
                </div>
              </section>

              {/* B. Biểu đồ phân bố cảm xúc */}
              <SentimentDistributionCharts posts={filteredPosts} />

              {/* C. Phân tích mức độ tương tác theo cảm xúc */}
              <EngagementAnalysisSection posts={filteredPosts} />

              {/* E. Danh sách bài đăng và kết quả dự đoán */}
              <PostsTable posts={filteredPosts} selectedModel={selectedModel} />

              {/* F. Dự đoán cảm xúc trực tiếp */}
              <LivePredictor
                mlSuite={mlSuite}
                defaultModel={selectedModel}
                vectorizerType={vectorizerType}
              />
            </>
          )}
        </main>
      </div>

      <StreamlitCodeDrawer
        open={codeDrawerOpen}
        onClose={() => setCodeDrawerOpen(false)}
      />
    </div>
  );
}
