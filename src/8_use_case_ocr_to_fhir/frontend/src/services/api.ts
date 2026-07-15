import axios from 'axios';

// Use Vite env variable if set, otherwise default to relative /api (proxied by Vite dev server)
const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

export const api = {
  // Prompt categories (from prompts.yaml)
  getPromptCategories: async (): Promise<{ categories: { key: string; value: string; label: string; hint: string }[] }> => {
    const response = await axios.get(`${API_BASE}/prompts/categories`);
    return response.data;
  },

  // Pigeon Mode - Simple extraction with streaming progress
  pigeonExtract: async (
    file: File,
    forceVlm: boolean = false,
    onEvent?: (evt: any) => void,
  ): Promise<{
    filename: string;
    medical_text: string;
    extraction_result: any;
    document_type?: string;
    error: string | null;
    has_text_layer?: boolean;
    used_vlm_ocr?: boolean;
  }> => {
    const form = new FormData();
    form.append('file', file);
    form.append('force_vlm', String(forceVlm));

    const res = await fetch(`${API_BASE}/simple/extract`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok && res.status !== 200) {
      throw new Error(`Pigeon extraction failed: ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error('Response body not readable');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalResult: any = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (!line) continue;
        try {
          const evt = JSON.parse(line);
          onEvent && onEvent(evt);
          // Store the complete event as final result
          if (evt.event === 'complete') {
            finalResult = evt;
          }
        } catch {}
      }
    }

    if (buffer.trim()) {
      try {
        const evt = JSON.parse(buffer.trim());
        onEvent && onEvent(evt);
        if (evt.event === 'complete') {
          finalResult = evt;
        }
      } catch {}
    }

    // Return the final result
    if (finalResult) {
      return {
        filename: finalResult.filename,
        medical_text: finalResult.medical_text,
        extraction_result: finalResult.extraction_result,
        document_type: finalResult.document_type,
        error: finalResult.error,
        has_text_layer: finalResult.has_text_layer,
        used_vlm_ocr: finalResult.used_vlm_ocr,
      };
    }

    throw new Error('No complete event received from server');
  },

  // FHIR Conversion
  convertToFHIR: async (extractionResult: any): Promise<{
    resources: any[];
    statistics: Record<string, number>;
  }> => {
    const response = await axios.post(`${API_BASE}/fhir/convert`, extractionResult);
    return response.data;
  },

  // Text inference - direct text input
  inferFromText: async (text: string): Promise<{
    success: boolean;
    medical_text: string;
    extraction_result: any;
    model: string;
  }> => {
    const response = await axios.post(`${API_BASE}/inference/text`, { text });
    return response.data;
  },

  // Compare Gemma vs Qwen extraction
  compareExtract: async (medicalText: string): Promise<{
    gemma_result: any;
    qwen_result: any;
  }> => {
    const response = await axios.post(`${API_BASE}/compare/extract`, {
      medical_text: medicalText,
    });
    return response.data;
  },

  // Showcase streaming
  showcaseProcess: async (
    file: File,
    category?: string,
    onEvent?: (evt: any) => void,
  ): Promise<void> => {
    const form = new FormData();
    form.append('file', file);
    if (category) form.append('category', category);

    const res = await fetch(`${API_BASE}/showcase/process`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok && res.status !== 200) {
      throw new Error(`Showcase process failed: ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) return;

    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (!line) continue;
        try {
          const evt = JSON.parse(line);
          onEvent && onEvent(evt);
        } catch {}
      }
    }

    if (buffer.trim()) {
      try {
        const evt = JSON.parse(buffer.trim());
        onEvent && onEvent(evt);
      } catch {}
    }
  },

  // Batch Classification
  getBatchFiles: async (): Promise<string[]> => {
    const response = await axios.get(`${API_BASE}/batch/files`);
    return response.data.files;
  },

  runBatchClassification: async (filename: string): Promise<any> => {
    const response = await axios.post(`${API_BASE}/batch/run`, { filename });
    return response.data;
  },
};
