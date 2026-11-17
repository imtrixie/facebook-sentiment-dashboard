import React, { useEffect, useState } from 'react';
import { Check, Copy, Download, FileCode2, X } from 'lucide-react';

interface StreamlitCodeDrawerProps {
  open: boolean;
  onClose: () => void;
}

type FileTab = 'app.py' | 'requirements.txt' | 'README.md';

const FILE_PATHS: Record<FileTab, string> = {
  'app.py': '/streamlit/app.py',
  'requirements.txt': '/streamlit/requirements.txt',
  'README.md': '/streamlit/README.md',
};

export const StreamlitCodeDrawer: React.FC<StreamlitCodeDrawerProps> = ({ open, onClose }) => {
  const [activeTab, setActiveTab] = useState<FileTab>('app.py');
  const [contents, setContents] = useState<Partial<Record<FileTab, string>>>({});
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open || contents[activeTab]) return;
    fetch(FILE_PATHS[activeTab])
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error('Không thể tải tệp'))))
      .then((text) => {
        setContents((prev) => ({ ...prev, [activeTab]: text }));
      })
      .catch(() => {
        setContents((prev) => ({
          ...prev,
          [activeTab]: `Không thể tải nội dung tệp ${activeTab}. Tệp đã được tạo sẵn trong thư mục gốc dự án.`,
        }));
      });
  }, [open, activeTab, contents]);

  if (!open) return null;

  const currentCode = contents[activeTab] ?? 'Đang tải mã nguồn...';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(currentCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownload = () => {
    const blob = new Blob([currentCode], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = activeTab;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-slate-900/40"
      role="dialog"
      aria-modal="true"
      aria-label="Mã nguồn Streamlit và Hướng dẫn Deploy"
    >
      <div className="flex h-full w-full max-w-3xl flex-col border-l border-slate-200 bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
          <div className="flex items-center gap-2.5">
            <FileCode2 className="h-5 w-5 text-blue-600" />
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                Mã nguồn Python Streamlit & Hướng dẫn GitHub / Render
              </h2>
              <p className="text-xs text-slate-500">
                Sẵn sàng chạy lệnh <code className="font-mono text-slate-700">streamlit run app.py</code> hoặc triển khai lên Render
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Đóng cửa sổ"
            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100 hover:text-slate-800"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-5 py-2.5">
          <div className="flex items-center gap-1.5">
            {(['app.py', 'requirements.txt', 'README.md'] as FileTab[]).map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveTab(tab)}
                className={`rounded-md px-3 py-1.5 font-mono text-xs font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-slate-900 text-white'
                    : 'text-slate-600 hover:bg-slate-200/70'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <a
              href="/streamlit/streamlit-facebook-sentiment.zip"
              download="streamlit-facebook-sentiment.zip"
              className="inline-flex items-center gap-1.5 rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700 whitespace-nowrap"
            >
              <Download className="h-3.5 w-3.5" />
              Tải trọn bộ (.zip)
            </a>
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? 'Đã sao chép' : 'Sao chép'}
            </button>
            <button
              type="button"
              onClick={handleDownload}
              className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
            >
              <Download className="h-3.5 w-3.5" />
              Tải {activeTab}
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto bg-slate-950 p-5">
          <pre className="font-mono text-xs leading-relaxed text-slate-100">
            <code>{currentCode}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
