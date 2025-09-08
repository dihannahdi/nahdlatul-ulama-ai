// Types for Nahdlatul Ulama AI

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface ChatHistory {
  messages: ChatMessage[];
}

export type IslamicMethod = 'bayani' | 'qiyasi' | 'istishlahi' | 'maqashidi';

export interface QuestionRequest {
  question: string;
  method?: IslamicMethod;
  chat_history?: ChatHistory;
  language?: 'id' | 'ar' | 'en';
}

export interface SourceReference {
  title: string;
  content: string;
  page?: string;
  author?: string;
  confidence: number;
}

export interface IslamicPrinciple {
  name: string;
  arabic: string;
  description: string;
  application: string;
}

export interface AnswerResponse {
  answer: string;
  sources: SourceReference[];
  method_used: IslamicMethod;
  confidence: number;
  islamic_principles: IslamicPrinciple[];
  reasoning_steps?: string[];
  alternative_views?: string[];
}

export interface MethodInfo {
  name: IslamicMethod;
  description: string;
  arabic: string;
}

export interface SystemStats {
  total_documents: number;
  available_methods: number;
  supported_languages: string[];
  islamic_principles: string[];
}

export interface SearchResult {
  content: string;
  source: string;
  metadata: Record<string, unknown>;
  relevance_score: number;
}
