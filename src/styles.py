import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Main Styling */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at top right, #1a237e, #0d1117 40%);
            color: #e6edf3;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(13, 17, 23, 0.95) !important;
            border-right: 1px solid rgba(88, 166, 255, 0.2);
            backdrop-filter: blur(15px);
        }

        /* Chat Message Styling */
        .stChatMessage {
            background-color: rgba(22, 27, 34, 0.6) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin-bottom: 1.2rem !important;
            border: 1px solid rgba(48, 54, 61, 0.5) !important;
            transition: all 0.3s ease;
        }

        .stChatMessage:hover {
            border-color: rgba(88, 166, 255, 0.4) !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        /* Header Styling */
        h1, h2, h3, h4 {
            background: linear-gradient(120deg, #58a6ff 0%, #bc8cf2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }

        /* Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.6rem 2.5rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            font-size: 0.8rem !important;
            letter-spacing: 1px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            box-shadow: 0 4px 15px rgba(35, 134, 54, 0.3) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(35, 134, 54, 0.5) !important;
        }

        /* Glassmorphism Panel */
        .glass-panel {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0d1117;
        }
        ::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #484f58;
        }

        /* Image hover effect */
        .stImage img {
            border-radius: 12px;
            transition: transform 0.3s ease;
        }
        .stImage img:hover {
            transform: scale(1.02);
        }
        /* Secondary / Clear Button */
        [data-testid="stSidebar"] div.stButton > button {
            background: linear-gradient(135deg, #ae1b1b, #8b1010) !important;
            box-shadow: 0 4px 15px rgba(174, 27, 27, 0.3) !important;
        }
        
        [data-testid="stSidebar"] div.stButton > button:hover {
            box-shadow: 0 8px 25px rgba(174, 27, 27, 0.5) !important;
        }
        </style>
    """, unsafe_allow_html=True)
