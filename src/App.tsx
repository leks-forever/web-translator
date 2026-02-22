import { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ModeToggle } from "@/components/mode-toggle"
import { Languages, ArrowRightLeft, Copy, Check, Loader2, Download } from 'lucide-react'

type ModelState = 'idle' | 'downloading' | 'ready'
type ProgressInfo = { loaded: number; total: number; percent: number }

const RING_RADIUS = 54;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

export default function App() {
  const [modelState, setModelState] = useState<ModelState>('idle');
  const [workerVersion, setWorkerVersion] = useState(0);
  const [sourceText, setSourceText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [sourceLang, setSourceLang] = useState('rus_Cyrl');
  const [targetLang, setTargetLang] = useState('lez_Cyrl');
  const [isTranslating, setIsTranslating] = useState(false);
  const [progressItems, setProgressItems] = useState<Record<string, ProgressInfo>>({});
  const [isCopied, setIsCopied] = useState(false);

  const worker = useRef<Worker | null>(null);

  const items = Object.values(progressItems);
  const totalLoaded = items.reduce((a, b) => a + b.loaded, 0);
  const totalBytes = items.reduce((a, b) => a + b.total, 0);
  const overallPercent = totalBytes > 0
    ? Math.min(100, (totalLoaded / totalBytes) * 100)
    : items.length > 0
      ? items.reduce((a, b) => a + b.percent, 0) / items.length
      : 0;
  const loadedMB = (totalLoaded / 1e6).toFixed(0);
  const totalMB = totalBytes > 0 ? (totalBytes / 1e6).toFixed(0) : null;
  const ringOffset = RING_CIRCUMFERENCE * (1 - overallPercent / 100);
  const hasProgress = totalLoaded > 0;

  useEffect(() => {
    worker.current = new Worker(new URL('./worker.ts?v=8', import.meta.url), {
      type: 'module'
    });

    worker.current.onmessage = (event) => {
      const { status, output, progress, error } = event.data;

      if (status === 'progress') {
        const { file, status: fileStatus, progress: fileProgressValue, loaded, total } = progress;
        if (fileStatus === 'initiate') {
          setProgressItems(prev => ({ ...prev, [file]: { loaded: 0, total: total ?? 0, percent: 0 } }));
        } else if (fileStatus === 'progress') {
          setProgressItems(prev => ({
            ...prev,
            [file]: { loaded: loaded ?? 0, total: total ?? 0, percent: fileProgressValue ?? 0 }
          }));
        } else if (fileStatus === 'done' || fileStatus === 'ready') {
          setProgressItems(prev => {
            const existing = prev[file];
            const t = total ?? existing?.total ?? 0;
            return { ...prev, [file]: { loaded: t, total: t, percent: 100 } };
          });
        }
      } else if (status === 'ready') {
        setModelState('ready');
      } else if (status === 'complete') {
        setTargetText(output);
        setIsTranslating(false);
      } else if (status === 'error') {
        setIsTranslating(false);
        setModelState('idle');
        console.error(error);
      }
    };

    return () => {
      worker.current?.terminate();
    };
  }, [workerVersion]);

  const handleLoadModel = useCallback(() => {
    if (!worker.current) return;
    setModelState('downloading');
    setProgressItems({});
    worker.current.postMessage({ type: 'load' });
  }, []);

  const handleCancel = useCallback(() => {
    setModelState('idle');
    setProgressItems({});
    setWorkerVersion(v => v + 1);
  }, []);

  const handleTranslate = useCallback(() => {
    if (!sourceText.trim() || !worker.current) return;
    setIsTranslating(true);
    worker.current.postMessage({
      text: sourceText,
      src_lang: sourceLang,
      tgt_lang: targetLang
    });
  }, [sourceText, sourceLang, targetLang]);

  const handleSwapLanguages = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setSourceText(targetText);
    setTargetText(sourceText);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(targetText);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      <div className="max-w-5xl mx-auto px-4 py-6 md:py-16 space-y-8 lg:space-y-12">

        {/* Header */}
        <header className="flex justify-between items-center border-b pb-4 lg:pb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary rounded-lg">
              <Languages className="w-5 h-5 lg:w-6 lg:h-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl lg:text-2xl font-bold tracking-tight text-foreground">LekTranslator</h1>
            </div>
          </div>
          <ModeToggle />
        </header>

        {/* Download gate */}
        {modelState !== 'ready' && (
          <div className="flex flex-col items-center justify-center py-16 gap-6 animate-in fade-in duration-500">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-semibold">Переводчик лезгинского языка</h2>
              <p className="text-muted-foreground text-sm">
                Перевод прямо в браузере — данные не покидают устройство.
              </p>
            </div>

            {/* Ring — always visible */}
            <div className="relative w-40 h-40">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                <circle
                  cx="60" cy="60" r={RING_RADIUS}
                  fill="none" strokeWidth="8"
                  className="stroke-muted"
                />
                {modelState === 'downloading' && (
                  <circle
                    cx="60" cy="60" r={RING_RADIUS}
                    fill="none" strokeWidth="8" strokeLinecap="round"
                    className="stroke-primary transition-all duration-500"
                    strokeDasharray={RING_CIRCUMFERENCE}
                    strokeDashoffset={ringOffset}
                  />
                )}
              </svg>

              <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
                {modelState === 'idle' && (
                  <button
                    onClick={handleLoadModel}
                    className="flex flex-col items-center gap-1.5 hover:opacity-70 active:scale-95 transition-all"
                  >
                    <Download className="w-7 h-7" />
                    <span className="text-xs font-medium leading-tight text-center">Скачать<br/>переводчик</span>
                  </button>
                )}

                {modelState === 'downloading' && (
                  <>
                    {hasProgress ? (
                      <>
                        <span className="text-2xl font-bold tabular-nums leading-none">{loadedMB}</span>
                        <span className="text-xs text-muted-foreground">
                          {totalMB ? `из ${totalMB} МБ` : 'МБ'}
                        </span>
                      </>
                    ) : (
                      <Loader2 className="w-7 h-7 animate-spin text-muted-foreground" />
                    )}
                  </>
                )}
              </div>
            </div>

            {modelState === 'downloading' && (
              <button
                onClick={handleCancel}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors underline underline-offset-2"
              >
                Отмена
              </button>
            )}
          </div>
        )}

        {/* Translator UI — only shown after model is ready */}
        {modelState === 'ready' && (
          <>
            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-4 lg:gap-6 items-center animate-in fade-in slide-in-from-bottom-4 duration-500">

              {/* Source Card */}
              <Card className="border-none shadow-none bg-transparent lg:bg-card lg:border lg:shadow-sm">
                <CardHeader className="px-4 py-3 lg:px-6 lg:py-4 border-b">
                  <div className="flex items-center h-8 font-medium px-2">
                    {sourceLang === 'rus_Cyrl' ? 'Русский' : 'Лезгинский'}
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  <Textarea
                    placeholder="Начните вводить текст..."
                    className="min-h-[120px] lg:min-h-[250px] border-none shadow-none focus-visible:ring-0 text-lg lg:text-xl p-4 lg:p-6 bg-transparent resize-none leading-relaxed"
                    value={sourceText}
                    onChange={(e) => setSourceText(e.target.value)}
                  />
                </CardContent>
              </Card>

              {/* Controls */}
              <div className="flex flex-row lg:flex-col items-center justify-center gap-2 lg:gap-4">
                <Button
                  variant="outline"
                  size="icon"
                  className="rounded-full h-9 w-9 lg:h-10 lg:w-10 shadow-sm transition-transform hover:rotate-180 duration-500"
                  onClick={handleSwapLanguages}
                >
                  <ArrowRightLeft className="w-4 h-4 lg:w-5 lg:h-5" />
                </Button>
                <Button
                  className="hidden lg:flex rounded-full h-14 w-14 shadow-lg active:scale-95 transition-all"
                  disabled={!sourceText.trim() || isTranslating}
                  onClick={handleTranslate}
                >
                  {isTranslating ? <Loader2 className="w-6 h-6 animate-spin" /> : <Languages className="w-6 h-6" />}
                </Button>
              </div>

              {/* Target Card */}
              <Card className="border-none shadow-none bg-transparent lg:bg-card lg:border lg:shadow-sm relative overflow-hidden">
                <CardHeader className="px-4 py-3 lg:px-6 lg:py-4 border-b flex flex-row items-center justify-between">
                  <div className="flex items-center h-8 font-medium px-2">
                    {targetLang === 'rus_Cyrl' ? 'Русский' : 'Лезгинский'}
                  </div>
                  {targetText && (
                    <Button variant="ghost" size="icon" className="h-8 w-8 ml-2" onClick={handleCopy}>
                      {isCopied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                    </Button>
                  )}
                </CardHeader>
                <CardContent className="p-0">
                  <div className="relative">
                    <Textarea
                      readOnly
                      placeholder="Перевод..."
                      className="min-h-[120px] lg:min-h-[250px] border-none shadow-none focus-visible:ring-0 text-lg lg:text-xl p-4 lg:p-6 bg-muted/30 lg:bg-transparent resize-none leading-relaxed"
                      value={targetText}
                    />
                    {isTranslating && (
                      <div className="absolute inset-0 flex items-center justify-center backdrop-blur-[1px] bg-background/10 transition-all">
                        <Loader2 className="w-8 h-8 lg:w-10 lg:h-10 animate-spin text-primary" />
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Mobile Action Button */}
            <div className="flex lg:hidden justify-center">
              <Button
                size="lg"
                className="w-full h-14 text-lg rounded-xl shadow-lg"
                disabled={!sourceText.trim() || isTranslating}
                onClick={handleTranslate}
              >
                {isTranslating ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Переводим...
                  </>
                ) : 'Перевести'}
              </Button>
            </div>
          </>
        )}

        {/* Footer */}
        <footer className="pt-8 border-t text-center text-xs text-muted-foreground space-y-2">
          <p>© 2026 LekTranslator</p>
          <p>
            Поддержать проект или предложить сотрудничество —{' '}
            <a
              href="https://t.me/lezgian_community"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-foreground transition-colors"
            >
              Напишите нам в Telegram
            </a>
          </p>
        </footer>
      </div>
    </div>
  )
}
