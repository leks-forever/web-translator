import { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { ModeToggle } from "@/components/mode-toggle"
import { Languages, ArrowRightLeft, Copy, Check, Loader2 } from 'lucide-react'

export default function App() {
  const [sourceText, setSourceText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [sourceLang, setSourceLang] = useState('rus_Cyrl');
  const [targetLang, setTargetLang] = useState('lez_Cyrl');
  const [isTranslating, setIsTranslating] = useState(false);
  const [progressItems, setProgressItems] = useState<Record<string, number>>({});
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const worker = useRef<Worker | null>(null);

  const totalProgress = Object.values(progressItems).length > 0
    ? Object.values(progressItems).reduce((a, b) => a + b, 0) / Object.values(progressItems).length
    : 0;

  useEffect(() => {
    // Add versioning to bypass worker cache
    worker.current = new Worker(new URL('./worker.ts?v=7', import.meta.url), {
      type: 'module'
    });

    worker.current.onmessage = (event) => {
      const { status, output, progress, error } = event.data;

      if (status === 'progress') {
        setIsModelLoading(true);
        const { file, status: fileStatus, progress: fileProgressValue } = progress;
        
        if (fileStatus === 'initiate') {
          setProgressItems(prev => ({ ...prev, [file]: 0 }));
        } else if (fileStatus === 'progress') {
          setProgressItems(prev => ({ ...prev, [file]: fileProgressValue }));
        } else if (fileStatus === 'done' || fileStatus === 'ready') {
          setProgressItems(prev => ({ ...prev, [file]: 100 }));
        }
      } else if (status === 'complete') {
        setTargetText(output);
        setIsTranslating(false);
        setIsModelLoading(false);
      } else if (status === 'error') {
        setIsTranslating(false);
        setIsModelLoading(false);
        console.error(error);
      }
    };

    return () => {
      worker.current?.terminate();
    };
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

        {/* Loading Progress */}
        {(isModelLoading || isTranslating) && totalProgress < 100 && (
          <div className="space-y-2 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className="flex justify-between text-xs lg:text-sm font-medium text-muted-foreground">
              <span>Загрузка...</span>
              <span>{Math.round(totalProgress)}%</span>
            </div>
            <Progress value={totalProgress} className="h-1" />
          </div>
        )}

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-4 lg:gap-6 items-center">
          
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

        {/* Footer */}
        <footer className="pt-8 border-t text-center text-xs text-muted-foreground">
          <p>© 2025 LekTranslator</p>
        </footer>
      </div>
    </div>
  )
}
