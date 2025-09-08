import { QuestionRequest, AnswerResponse, MethodInfo, SystemStats, SearchResult } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async askQuestion(request: QuestionRequest): Promise<AnswerResponse> {
    return this.request<AnswerResponse>('/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getMethods(): Promise<{ methods: MethodInfo[] }> {
    return this.request<{ methods: MethodInfo[] }>('/methods');
  }

  async getStats(): Promise<SystemStats> {
    return this.request<SystemStats>('/stats');
  }

  async search(query: string, limit = 5): Promise<{ query: string; results: SearchResult[] }> {
    const params = new URLSearchParams({ query, limit: limit.toString() });
    return this.request<{ query: string; results: SearchResult[] }>(`/search?${params}`);
  }

  async healthCheck(): Promise<{ status: string; [key: string]: any }> {
    return this.request<{ status: string; [key: string]: any }>('/health');
  }
}

export const apiService = new ApiService();
