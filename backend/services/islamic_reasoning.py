from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from models.schemas import IslamicMethod, SourceReference, IslamicPrinciple

class IslamicReasoningEngine:
    """Islamic jurisprudence reasoning engine using NU methodology"""
    
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Islamic principles (Fikrah Nahdliyah)
        self.nu_principles = [
            {
                "name": "Tawassuth",
                "arabic": "التوسط", 
                "description": "Moderation and balanced approach"
            },
            {
                "name": "Tasamuh", 
                "arabic": "التسامح",
                "description": "Tolerance and acceptance of differences"
            },
            {
                "name": "Tawazun",
                "arabic": "التوازن", 
                "description": "Balance between spiritual and worldly matters"
            },
            {
                "name": "I'tidal",
                "arabic": "الاعتدال",
                "description": "Justice and righteousness"
            }
        ]
    
    async def process_question(
        self, 
        question: str, 
        method: IslamicMethod = IslamicMethod.BAYANI,
        chat_history=None
    ) -> Dict[str, Any]:
        """Process Islamic jurisprudence question using specified method"""
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.invoke(question)
        
        # Apply Islamic reasoning method
        if method == IslamicMethod.BAYANI:
            result = await self._apply_bayani_method(question, relevant_docs)
        elif method == IslamicMethod.QIYASI:
            result = await self._apply_qiyasi_method(question, relevant_docs)
        elif method == IslamicMethod.ISTISHLAHI:
            result = await self._apply_istishlahi_method(question, relevant_docs)
        elif method == IslamicMethod.MAQASHIDI:
            result = await self._apply_maqashidi_method(question, relevant_docs)
        else:
            result = await self._apply_bayani_method(question, relevant_docs)
        
        return result
    
    async def _apply_bayani_method(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        """Apply Bayani method - textual analysis from Quran and Sunnah"""
        
        context = self._format_context(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        Anda adalah ulama Nahdlatul Ulama yang ahli dalam metode Istinbath Al-Ahkam. 
        Gunakan metode Bayani (البياني) untuk menjawab pertanyaan fiqh berikut.

        Metode Bayani adalah pengambilan hukum dari nash (Al-Qur'an dan As-Sunnah) dengan langkah:
        1. Mengkaji sabab al-nuzul/wurud (sebab turunnya ayat/hadits)
        2. Memahami makna lafal dan konteks
        3. Menerapkan kaidah bahasa Arab dan ushul fiqh
        4. Mempertimbangkan pandangan madzhab empat (Hanafi, Maliki, Syafi'i, Hambali)

        Konteks dari kitab-kitab Islam:
        {context}

        Pertanyaan: {question}

        Jawab dengan menggunakan prinsip-prinsip NU:
        - Tawassuth (التوسط): Sikap moderat, tidak ekstrem
        - Tasamuh (التسامح): Toleran terhadap perbedaan pendapat
        - Tawazun (التوازن): Seimbang antara dunia dan akhirat  
        - I'tidal (الاعتدال): Adil dan lurus

        Berikan jawaban yang:
        1. Berdasarkan dalil nash yang kuat
        2. Mengutip sumber yang relevan
        3. Menerapkan prinsip-prinsip NU
        4. Praktis dan dapat diamalkan
        """)
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = await chain.ainvoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "sources": self._extract_sources(docs),
            "method_used": IslamicMethod.BAYANI,
            "confidence": 0.85,
            "islamic_principles": self._get_applied_principles(["tawassuth", "tasamuh"])
        }
    
    async def _apply_qiyasi_method(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        """Apply Qiyasi method - analogical reasoning"""
        
        context = self._format_context(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        Gunakan metode Qiyasi (القياسي) untuk menjawab pertanyaan fiqh ini.

        Metode Qiyasi adalah analogical reasoning dengan unsur:
        1. Ashl (الأصل): Kasus yang sudah ada nashnya
        2. Far' (الفرع): Kasus baru yang ditanyakan  
        3. 'Illah (العلة): Sebab/alasan hukum yang menghubungkan
        4. Hukm (الحكم): Hukum yang ditetapkan

        Konteks: {context}
        Pertanyaan: {question}

        Jawab dengan:
        1. Identifikasi kasus serupa (ashl) yang ada nashnya
        2. Temukan 'illah (alasan hukum) yang sama
        3. Terapkan qiyas dengan hati-hati
        4. Pertimbangkan madzhab yang mu'tabar
        5. Gunakan prinsip NU: moderat dan toleran
        """)
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt  
            | self.llm
            | StrOutputParser()
        )
        
        answer = await chain.ainvoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "sources": self._extract_sources(docs),
            "method_used": IslamicMethod.QIYASI,
            "confidence": 0.80,
            "islamic_principles": self._get_applied_principles(["tawazun", "i'tidal"])
        }
    
    async def _apply_istishlahi_method(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        """Apply Istishlahi method - public interest consideration"""
        
        context = self._format_context(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        Gunakan metode Istishlahi (الاستصلاحي) untuk pertanyaan ini.

        Metode Istishlahi mempertimbangkan maslahat (kemaslahatan) dengan:
        1. Maslahat yang sesuai dengan maqashid syariah
        2. Tidak bertentangan dengan nash qath'i
        3. Mempertimbangkan kondisi zaman dan tempat
        4. Mengutamakan kemaslahatan umum

        Konteks: {context}
        Pertanyaan: {question}

        Analisis dengan:
        1. Identifikasi maslahat dan mafsadat
        2. Timbang kemanfaatan dan kemudharatan
        3. Pertimbangkan konteks Indonesia/Nusantara
        4. Tetap dalam koridor syariah yang mu'tabar
        5. Terapkan prinsip NU yang moderat dan toleran
        """)
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = await chain.ainvoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "sources": self._extract_sources(docs),
            "method_used": IslamicMethod.ISTISHLAHI,
            "confidence": 0.75,
            "islamic_principles": self._get_applied_principles(["tawassuth", "tawazun"])
        }
    
    async def _apply_maqashidi_method(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        """Apply Maqashidi method - objectives of Sharia"""
        
        context = self._format_context(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        Gunakan metode Maqashidi (المقاصدي) berdasarkan maqashid syariah.

        Lima maqashid syariah:
        1. Hifz al-Din (حفظ الدين): Menjaga agama
        2. Hifz al-Nafs (حفظ النفس): Menjaga jiwa
        3. Hifz al-Aql (حفظ العقل): Menjaga akal
        4. Hifz al-Nasl (حفظ النسل): Menjaga keturunan
        5. Hifz al-Mal (حفظ المال): Menjaga harta

        Konteks: {context}
        Pertanyaan: {question}

        Analisis berdasarkan:
        1. Mana maqashid yang terkait dengan pertanyaan
        2. Prioritas maqashid (dharuriyyat, hajiyyat, tahsiniyyat)
        3. Keseimbangan antar maqashid
        4. Konteks kekinian yang relevan
        5. Prinsip NU: moderat, toleran, seimbang, dan adil
        """)
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = await chain.ainvoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "sources": self._extract_sources(docs),
            "method_used": IslamicMethod.MAQASHIDI,
            "confidence": 0.85,
            "islamic_principles": self._get_applied_principles(["tawassuth", "tawazun", "i'tidal"])
        }
    
    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, doc in enumerate(docs[:5]):  # Limit to top 5
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content[:500]  # Limit content length
            context_parts.append(f"[Sumber {i+1}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, docs: List[Document]) -> List[SourceReference]:
        """Extract source references from documents"""
        sources = []
        for doc in docs[:3]:  # Top 3 sources
            source = SourceReference(
                title=doc.metadata.get("source", "Unknown"),
                content=doc.page_content[:200] + "...",
                page=doc.metadata.get("page"),
                confidence=0.8  # Placeholder confidence
            )
            sources.append(source)
        
        return sources
    
    def _get_applied_principles(self, principle_names: List[str]) -> List[IslamicPrinciple]:
        """Get Islamic principles that were applied"""
        principles = []
        for name in principle_names:
            for p in self.nu_principles:
                if p["name"].lower() == name.lower():
                    principle = IslamicPrinciple(
                        name=p["name"],
                        arabic=p["arabic"],
                        description=p["description"],
                        application=f"Applied in reasoning process"
                    )
                    principles.append(principle)
                    break
        
        return principles
