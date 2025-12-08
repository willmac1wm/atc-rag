from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import PyPDF2, numpy as np, os
from pathlib import Path
from docx import Document
from pptx import Presentation

app = Flask(__name__)
CORS(app)

class RAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = []
        self.emb = None
        self.cmap = []
    
    def load_docs(self, folder):
        for f in Path(folder).glob('*.pdf'):
            try:
                txt = ''.join(p.extract_text() for p in PyPDF2.PdfReader(open(f,'rb')).pages)
                if len(txt) > 100:
                    self.docs.append({'name': f.name, 'text': txt})
            except: pass
        
        for f in Path(folder).glob('*.docx'):
            try:
                txt = '\n'.join(p.text for p in Document(f).paragraphs)
                if len(txt) > 100:
                    self.docs.append({'name': f.name, 'text': txt})
            except: pass
        
        chunks = []
        for i, d in enumerate(self.docs):
            for j in range(0, len(d['text'].split()), 500):
                chunk = ' '.join(d['text'].split()[j:j+500])
                chunks.append(chunk)
                self.cmap.append((i, chunk))
        
        self.emb = self.model.encode(chunks, show_progress_bar=False)

rag = RAG()
docs_path = os.getenv('DOCS_PATH', '/app/training_docs')
if os.path.exists(docs_path):
    rag.load_docs(docs_path)

API_KEY = os.getenv('GEMINI_API_KEY', '')
if API_KEY:
    genai.configure(api_key=API_KEY)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'docs_loaded': len(rag.docs)})

@app.route('/search', methods=['POST'])
def search():
    question = request.json.get('question', '')
    qe = rag.model.encode([question])[0]
    sims = np.dot(rag.emb, qe)
    top = np.argsort(sims)[::-1][:3]
    
    context = []
    sources = []
    for idx in top:
        di, chunk = rag.cmap[idx]
        doc = rag.docs[di]
        context.append({'source': doc['name'], 'text': chunk[:800]})
        sources.append(doc['name'])
    
    if API_KEY:
        ctx_text = '\n\n'.join(f"[{c['source']}]\n{c['text']}" for c in context)
        prompt = f"FAA Materials:\n{ctx_text}\n\nQuestion: {question}\n\nAnswer as FAA instructor:"
        
        gmodel = genai.GenerativeModel('gemini-pro')
        resp = gmodel.generate_content(prompt)
        
        return jsonify({'answer': resp.text, 'sources': sources, 'context': context})
    
    return jsonify({'context': context, 'sources': sources})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
