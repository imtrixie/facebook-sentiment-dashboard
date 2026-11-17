import React, { useState } from 'react';
import { Sparkles } from 'lucide-react';
import {
  ModelType,
  PredictionOutput,
  Sentiment,
  VectorizerType,
} from '../types/dashboard';
import { SENTIMENT_CLASSES, TrainedMlSuite } from '../utils/mlEngine';
import { SENTIMENT_META } from './Charts';

interface LivePredictorProps {
  mlSuite: TrainedMlSuite;
  defaultModel: ModelType;
  vectorizerType: VectorizerType;
}

const SAMPLE_POSTS = [
  {
    label: 'Mẫu Tích cực',
    text: 'Absolutely love this new product update! Works smoothly and saved my day 🎉',
  },
  {
    label: 'Mẫu Tiêu cực',
    text: 'Having trouble logging into my account again, support is not replying. Very frustrating!',
  },
  {
    label: 'Mẫu Trung lập',
    text: 'In-depth: What the quarterly market report means for New York investors this week.',
  },
];

export const LivePredictor: React.FC<LivePredictorProps> = ({
  mlSuite,
  defaultModel,
  vectorizerType,
}) => {
  const [text, setText] = useState(SAMPLE_POSTS[0].text);
  const [modelName, setModelName] = useState<ModelType>(defaultModel);
  const [result, setResult] = useState<PredictionOutput | null>(() =>
    mlSuite.predictText(defaultModel, SAMPLE_POSTS[0].text)
  );

  const nbEval = mlSuite.evaluations['Naive Bayes'];
  const dtEval = mlSuite.evaluations['Decision Tree'];

  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = text.trim();
    if (!cleaned) {
      setResult(null);
      return;
    }
    setResult(mlSuite.predictText(modelName, cleaned));
  };

  return (
    <section aria-labelledby="section-live-ml" className="space-y-4">
      <div className="border-b border-slate-200 pb-2">
        <h2 id="section-live-ml" className="text-base font-semibold text-slate-900">
          F. Dự đoán cảm xúc bài đăng mới (Machine Learning)
        </h2>
        <p className="text-sm text-slate-500">
          Huấn luyện bằng Train/Test Split (80%/20%) với trích xuất đặc trưng{' '}
          <span className="font-semibold text-slate-700">{vectorizerType}</span>. Đánh giá độc lập trên tập kiểm thử (Test set n ={' '}
          <span className="tabular-nums">{nbEval.testSize.toLocaleString('vi-VN')}</span>).
        </p>
      </div>

      {/* Hiệu năng hai mô hình trên tập Test */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="rounded-lg border border-slate-200 bg-white p-4 lg:col-span-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Naive Bayes · {vectorizerType}
          </div>
          <div className="mt-1.5 text-2xl font-bold text-slate-900 tabular-nums">
            {(nbEval.testAccuracy * 100).toFixed(2)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 tabular-nums">
            Accuracy trên Test set · Macro F1: {(nbEval.testMacroF1 * 100).toFixed(2)}%
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4 lg:col-span-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Decision Tree · {vectorizerType}
          </div>
          <div className="mt-1.5 text-2xl font-bold text-slate-900 tabular-nums">
            {(dtEval.testAccuracy * 100).toFixed(2)}%
          </div>
          <div className="mt-1 text-xs text-slate-500 tabular-nums">
            Accuracy trên Test set · Macro F1: {(dtEval.testMacroF1 * 100).toFixed(2)}%
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4 lg:col-span-6 overflow-x-auto">
          <table className="w-full text-left text-xs tabular-nums">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="pb-1.5 font-semibold">Mô hình (Test Split 20%)</th>
                <th className="pb-1.5 text-right font-semibold">Accuracy</th>
                <th className="pb-1.5 text-right font-semibold">F1 Positive</th>
                <th className="pb-1.5 text-right font-semibold">F1 Neutral</th>
                <th className="pb-1.5 text-right font-semibold">F1 Negative</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {[nbEval, dtEval].map((ev) => (
                <tr key={ev.modelName}>
                  <td className="py-2 font-semibold text-slate-800">{ev.modelName}</td>
                  <td className="py-2 text-right font-medium text-slate-900">
                    {(ev.testAccuracy * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 text-right text-emerald-700">
                    {(ev.classMetrics.positive.f1 * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 text-right text-amber-700">
                    {(ev.classMetrics.neutral.f1 * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 text-right text-red-700">
                    {(ev.classMetrics.negative.f1 * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Khu vực nhập liệu và kết quả dự đoán trực tiếp */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <form
          onSubmit={handlePredict}
          className="space-y-3 rounded-lg border border-slate-200 bg-white p-5 lg:col-span-7"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <label htmlFor="live-post-input" className="text-sm font-semibold text-slate-800">
              Nhập nội dung bài đăng Facebook mới
            </label>
            <div className="flex flex-wrap items-center gap-1.5">
              {SAMPLE_POSTS.map((sample) => (
                <button
                  key={sample.label}
                  type="button"
                  onClick={() => {
                    setText(sample.text);
                    setResult(mlSuite.predictText(modelName, sample.text));
                  }}
                  className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-100"
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          <textarea
            id="live-post-input"
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Nhập nội dung bài đăng bằng tiếng Anh hoặc ngôn ngữ trong tập dữ liệu..."
            className="w-full rounded-md border border-slate-300 bg-white p-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none"
          />

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-4 text-sm text-slate-700">
              <span className="font-medium text-slate-600">Mô hình:</span>
              {(['Naive Bayes', 'Decision Tree'] as ModelType[]).map((m) => (
                <label key={m} className="inline-flex cursor-pointer items-center gap-1.5">
                  <input
                    type="radio"
                    name="live-model"
                    value={m}
                    checked={modelName === m}
                    onChange={() => {
                      setModelName(m);
                      if (text.trim()) {
                        setResult(mlSuite.predictText(m, text.trim()));
                      }
                    }}
                    className="h-4 w-4 text-blue-600"
                  />
                  <span className="font-medium">{m}</span>
                </label>
              ))}
            </div>

            <button
              type="submit"
              disabled={!text.trim()}
              className="inline-flex items-center justify-center gap-1.5 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Sparkles className="h-4 w-4" />
              Dự đoán cảm xúc
            </button>
          </div>
        </form>

        <div className="rounded-lg border border-slate-200 bg-white p-5 lg:col-span-5">
          <h3 className="text-sm font-semibold text-slate-800">
            Kết quả & Xác suất dự đoán ({modelName})
          </h3>

          {result ? (
            <div className="mt-4 space-y-4">
              <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-medium text-slate-500">
                  Nhãn cảm xúc dự đoán
                </div>
                <div
                  className="mt-1 text-xl font-bold"
                  style={{ color: SENTIMENT_META[result.label].color }}
                >
                  {SENTIMENT_META[result.label].label}
                </div>
                <div className="mt-1 text-xs text-slate-600 tabular-nums">
                  Độ tin cậy:{' '}
                  <span className="font-semibold text-slate-900">
                    {(result.probabilities[result.label] * 100).toFixed(2)}%
                  </span>
                </div>
              </div>

              <div className="space-y-2.5">
                {SENTIMENT_CLASSES.map((cls: Sentiment) => {
                  const pct = result.probabilities[cls] * 100;
                  const meta = SENTIMENT_META[cls];
                  return (
                    <div key={cls} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-slate-700">{meta.label}</span>
                        <span className="font-semibold text-slate-900 tabular-nums">
                          {pct.toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2.5 w-full overflow-hidden rounded-sm bg-slate-100">
                        <div
                          className="h-full transition-all duration-200"
                          style={{
                            width: `${Math.max(pct, 1)}%`,
                            backgroundColor: meta.color,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="mt-8 text-center text-sm text-slate-500">
              Vui lòng nhập nội dung bài đăng và nhấn Dự đoán cảm xúc.
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
