'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, BookOpen, Brain, Scale, Heart, Loader2 } from 'lucide-react';
import { apiService } from '@/lib/api';
import { ChatMessage, IslamicMethod, MethodInfo } from '@/types';
import { cn } from '@/lib/utils';

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState<IslamicMethod>('bayani');
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load available methods
    apiService.getMethods().then(response => {
      if (response && response.methods && Array.isArray(response.methods)) {
        setMethods(response.methods);
      }
    }).catch(error => {
      console.error('Failed to load methods:', error);
      // Set default methods if API fails
      setMethods([
        {
          name: 'bayani',
          description: 'Metode interpretasi teks berbasis dalil Al-Quran dan Hadits',
          arabic: 'البياني'
        },
        {
          name: 'istislahi',
          description: 'Metode pertimbangan kemaslahatan untuk kebaikan umum',
          arabic: 'الاستصلاحي'
        },
        {
          name: 'taqrir',
          description: 'Metode berdasarkan konsensus ulama',
          arabic: 'التقريري'
        }
      ]);
    });

    // Add welcome message
    setMessages([{
      role: 'assistant',
      content: 'السلام عليكم ورحمة الله وبركاته\n\nSelamat datang di Nahdlatul Ulama AI - Asisten Fiqh berbasis metodologi NU.\n\nSaya siap membantu Anda dengan pertanyaan-pertanyaan fiqh menggunakan prinsip-prinsip Ahlussunnah wal Jama\'ah an-Nahdliyyah.',
      timestamp: new Date().toISOString()
    }]);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await apiService.askQuestion({
        question: input,
        method: selectedMethod,
        chat_history: { messages: messages },
        language: 'id'
      });

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const getMethodIcon = (method: IslamicMethod) => {
    switch (method) {
      case 'bayani': return <BookOpen className="w-4 h-4" />;
      case 'qiyasi': return <Brain className="w-4 h-4" />;
      case 'istishlahi': return <Heart className="w-4 h-4" />;
      case 'maqashidi': return <Scale className="w-4 h-4" />;
      default: return <BookOpen className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-600 rounded-lg flex items-center justify-center">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Nahdlatul Ulama AI</h1>
              <p className="text-sm text-gray-600">Asisten Fiqh Ahlussunnah wal Jama&apos;ah</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="max-w-4xl mx-auto px-4 py-6">
        <div className="bg-white rounded-lg shadow-lg h-[600px] flex flex-col">
          
          {/* Method Selector */}
          <div className="p-4 border-b bg-gray-50 rounded-t-lg">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Metode Istinbath:
            </label>
            <div className="flex gap-2 flex-wrap">
              {methods && methods.length > 0 && methods.map((method) => (
                <button
                  key={method.name}
                  onClick={() => setSelectedMethod(method.name)}
                  className={cn(
                    "px-3 py-2 rounded-md text-sm font-medium flex items-center gap-2 transition-colors",
                    selectedMethod === method.name
                      ? "bg-emerald-600 text-white"
                      : "bg-white text-gray-700 hover:bg-gray-100 border"
                  )}
                >
                  {getMethodIcon(method.name)}
                  <span>{method.arabic}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={cn(
                  "flex",
                  message.role === 'user' ? "justify-end" : "justify-start"
                )}
              >
                <div
                  className={cn(
                    "max-w-[80%] rounded-lg px-4 py-3",
                    message.role === 'user'
                      ? "bg-emerald-600 text-white"
                      : "bg-gray-100 text-gray-900"
                  )}
                >
                  <div className="whitespace-pre-wrap text-sm leading-relaxed">
                    {message.content}
                  </div>
                  {message.timestamp && (
                    <div className={cn(
                      "text-xs mt-2 opacity-70",
                      message.role === 'user' ? "text-emerald-100" : "text-gray-500"
                    )}>
                      {new Date(message.timestamp).toLocaleTimeString('id-ID')}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg px-4 py-3 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-gray-600" />
                  <span className="text-sm text-gray-600">Sedang memproses...</span>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="p-4 border-t">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Tanyakan pertanyaan fiqh Anda..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className={cn(
                  "px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-colors",
                  "bg-emerald-600 text-white hover:bg-emerald-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
                )}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                Kirim
              </button>
            </div>
          </form>
        </div>

        {/* NU Principles */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { name: 'Tawassuth', arabic: 'التوسط', desc: 'Moderat', color: 'emerald' },
            { name: 'Tasamuh', arabic: 'التسامح', desc: 'Toleran', color: 'blue' },
            { name: 'Tawazun', arabic: 'التوازن', desc: 'Seimbang', color: 'purple' },
            { name: 'I\'tidal', arabic: 'الاعتدال', desc: 'Adil', color: 'orange' }
          ].map((principle) => (
            <div key={principle.name} className="bg-white rounded-lg p-4 shadow-sm text-center">
              <div className="text-lg font-semibold text-gray-900">{principle.arabic}</div>
              <div className="text-sm font-medium text-gray-700">{principle.name}</div>
              <div className="text-xs text-gray-500 mt-1">{principle.desc}</div>
            </div>
          ))}
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-gray-600">
        <p>Nahdlatul Ulama AI - Dikembangkan untuk kepentingan pendidikan dan dakwah</p>
        <p className="mt-1">بارك الله فيكم</p>
      </footer>
    </div>
  );
}
