import { pipeline, env, type PipelineType } from '@huggingface/transformers';

// Skip local check
env.allowLocalModels = false;

// Most stable and widely used model ID for NLLB-200 in transformers.js
const MODEL_ID = 'Xenova/nllb-200-distilled-600m';

class TranslationPipeline {
    static task: PipelineType = 'translation';
    static model = MODEL_ID;
    static instance: any = null;

    static async getInstance(progress_callback?: (progress: any) => void) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { 
                progress_callback,
                device: 'wasm', // Using WASM for maximum stability like the official Xenova demo
                dtype: 'q8',    // q8 is the perfect balance: ~600MB and high quality
            });
        }
        return this.instance;
    }
}

self.onmessage = async (event) => {
    const { text, src_lang, tgt_lang } = event.data;

    try {
        const translator = await TranslationPipeline.getInstance((progress: any) => {
            self.postMessage({ status: 'progress', progress });
        });

        const output = await translator(text, {
            src_lang,
            tgt_lang,
        }) as any;

        self.postMessage({
            status: 'complete',
            output: output[0].translation_text
        });
    } catch (error: any) {
        const errorMessage = error?.message || error?.toString() || 'Unknown error';
        console.error('Worker error details:', error);
        
        // Fallback logic
        if (errorMessage.includes('WebGPU') || errorMessage.includes('Aborted') || typeof error === 'number' || errorMessage.includes('Unauthorized')) {
            console.warn('GPU/Access failed, falling back to CPU (WASM)...');
            try {
                TranslationPipeline.instance = null;
                const cpuTranslator: any = await pipeline(TranslationPipeline.task, TranslationPipeline.model, {
                    device: 'wasm',
                    dtype: 'q8', // Use q8 on CPU for better quality and stability
                    progress_callback: (progress: any) => {
                        self.postMessage({ status: 'progress', progress });
                    },
                });
                
                const output = await cpuTranslator(text, { 
                    src_lang, 
                    tgt_lang,
                    max_new_tokens: 256
                });
                self.postMessage({ status: 'complete', output: output[0].translation_text });
            } catch (cpuError: any) {
                console.error('CPU fallback failed:', cpuError);
                self.postMessage({ status: 'error', error: cpuError?.message || 'CPU Fallback failed' });
            }
        } else {
            self.postMessage({ status: 'error', error: errorMessage });
        }
    }
};
