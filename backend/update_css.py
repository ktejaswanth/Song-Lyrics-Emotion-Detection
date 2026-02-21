import codecs
import re

file_path = 'streamlit_app.py'

with codecs.open(file_path, 'r', 'utf-8') as f:
    text = f.read()

# 1. Add components import
if 'import streamlit.components.v1 as components' not in text:
    text = text.replace('import speech_recognition as sr', 'import speech_recognition as sr\nimport streamlit.components.v1 as components')

# 2. Replace style tag
css_start = text.find('<style>')
css_end = text.find('</style>') + len('</style>')

new_css = """<style>
    /* ========================================= */
    /* 1. GLOBAL VARIABLES & BASE STYLES         */
    /* ========================================= */
    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ========================================= */
    /* 2. LAYOUT COMPONENTS                      */
    /* ========================================= */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ========================================= */
    /* 3. METRICS & BADGES                       */
    /* ========================================= */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: auto;
    }

    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin: 0.5rem;
    }
    
    .emotion-joy { background: linear-gradient(135deg, #f6d365, #fda085); color: #333; }
    .emotion-sadness { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .emotion-anger { background: linear-gradient(135deg, #f5576c, #ff6b6b); color: white; }
    .emotion-fear { background: linear-gradient(135deg, #4facfe, #00f2fe); color: #333; }
    .emotion-surprise { background: linear-gradient(135deg, #43e97b, #38f9d7); color: #333; }
    .emotion-love { background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }

    /* ========================================= */
    /* 4. STREAMLIT ELEMENTS OVERRIDES           */
    /* ========================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #a0aec0;
        padding: 10px 20px;
        white-space: nowrap;
        margin-bottom: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* ========================================= */
    /* 5. RESPONSIVE MEDIA QUERIES               */
    /* ========================================= */
    @media screen and (max-width: 768px) {
        .main-header {
            padding: 1.5rem 1rem;
        }
        .main-header h1 {
            font-size: 1.8rem;
        }
        .main-header p {
            font-size: 0.95rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
        .glass-card {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .emotion-badge {
            font-size: 1rem;
            padding: 0.4rem 1.2rem;
        }
        .stButton > button {
            padding: 0.6rem 1rem !important;
            font-size: 0.9rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px;
            font-size: 0.8rem;
        }
        div[data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 1rem;
        }
    }
</style>"""

if css_start != -1 and css_end != -1:
    text = text[:css_start] + new_css + text[css_end:]

# 3. Inject PWA Script
pwa_code = """
# ============================================================
# PROGRESSIVE WEB APP (PWA) REGISTRATION
# ============================================================
pwa_script = '''
<script>
    if (window.parent.navigator.serviceWorker) {
        const manifest = {
            "name": "🎵 Song Lyrics Emotion Detection AI",
            "short_name": "MoodMate AI",
            "description": "Analyze the emotional fingerprint of any song.",
            "start_url": "./",
            "display": "standalone",
            "background_color": "#0f0c29",
            "theme_color": "#667eea",
            "icons": [
                {
                    "src": "https://cdn-icons-png.flaticon.com/512/3048/3048122.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "https://cdn-icons-png.flaticon.com/512/3048/3048122.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        };
        const manifestBlob = new Blob([JSON.stringify(manifest)], {type: 'application/json'});
        const manifestUrl = URL.createObjectURL(manifestBlob);
        
        let link = window.parent.document.querySelector('link[rel="manifest"]');
        if (!link) {
            link = window.parent.document.createElement('link');
            link.rel = 'manifest';
            window.parent.document.head.appendChild(link);
        }
        link.href = manifestUrl;

        const sw = `
            self.addEventListener('install', e => {
                console.log('PWA Service Worker Install');
            });
            self.addEventListener('fetch', e => {
                const url = new URL(e.request.url);
                if (url.pathname.includes('/_stcore/')) {
                    // Do not cache websockets or live Streamlit data
                    return fetch(e.request);
                }
                
                // Network-first strategy for frontend assets
                e.respondWith(
                    fetch(e.request).catch(function() {
                        return caches.match(e.request);
                    })
                );
            });
        `;
        const blob = new Blob([sw], {type: 'application/javascript'});
        const url = URL.createObjectURL(blob);
        
        window.parent.navigator.serviceWorker.register(url)
            .then(function(r) { console.log('PWA SW registered!'); })
            .catch(function(e) { console.log('SW error!', e); });
    }
</script>
'''
components.html(pwa_script, height=0, width=0)
"""

if 'PROGRESSIVE WEB APP (PWA) REGISTRATION' not in text:
    css_section_end = text.find('""", unsafe_allow_html=True)') + len('""", unsafe_allow_html=True)')
    text = text[:css_section_end] + '\n' + pwa_code + '\n' + text[css_section_end:]

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(text)

print('Updated successfully')
