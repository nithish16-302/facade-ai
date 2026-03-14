"use client";

import { useState, useRef, useEffect } from "react";

const PALETTES = [
  { id: "nordic", name: "Nordic Noir", desc: "Matte Onyx & Anodized Aluminum", gradient: "var(--palette-nordic)" },
  { id: "biophilic", name: "Biophilic Greens", desc: "Sage Shadows & Raw Corten", gradient: "var(--palette-biophilic)" },
  { id: "brutalist", name: "Eco Brutalism", desc: "Board-formed Concrete & Oak", gradient: "var(--palette-brutalist)" },
  { id: "monolith", name: "Desert Monolith", desc: "Terracotta Plaster & Brushed Brass", gradient: "var(--palette-monolith)" },
  { id: "americana", name: "New Americana", desc: "Deep Navy & Alabaster Whites", gradient: "var(--palette-americana)" },
  { id: "haveli", name: "Haveli Sandstone", desc: "Warm Rajasthani Ochre & Carved Jali", gradient: "linear-gradient(135deg, #c8874a, #e8c07d)" },
  { id: "kerala", name: "Kerala Verdant", desc: "Deep Laterite Red & Tropical Teak", gradient: "linear-gradient(135deg, #7a2e1a, #3d7a4a)" },
  { id: "mughal", name: "Mughal Marble", desc: "Creamy Makrana White & Inlaid Pietra Dura", gradient: "linear-gradient(135deg, #f5f0e8, #b8a99a)" }
];

export default function Home() {
  const [selectedPalette, setSelectedPalette] = useState("nordic");
  const [isHovering, setIsHovering] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  
  // States: 'idle', 'analyzing', 'generating', 'done'
  const [appState, setAppState] = useState<'idle' | 'analyzing' | 'generating' | 'done'>('idle');
  
  // Toggle between original and generated view
  const [showGenerated, setShowGenerated] = useState(false);

  // State for the uploaded file object specifically for the backend form data
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  // Handle Upload
  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const fileUrl = URL.createObjectURL(e.target.files[0]);
      setUploadedImage(fileUrl);
      setUploadFile(e.target.files[0]);
      setAppState('idle');
    }
  };

  // Generate Process hitting actual FastAPI Backend
  const handleGenerate = async () => {
    if (!uploadFile) return;

    setAppState('analyzing');
    
    try {
      const formData = new FormData();
      formData.append('image', uploadFile);
      formData.append('palette_id', selectedPalette);

      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${API_URL}/api/v1/generate`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to generate facade");
      }

      setAppState('generating');
      
      const data = await res.json();
      console.log("VISION Engine Output:", data.vision_analysis);
      
      // Store the generated image separately — keep original for the slider
      if (data.generated_image_url) {
        setGeneratedImage(data.generated_image_url);
        setShowGenerated(true); // Auto-reveal the generated result
        setAppState('done');
      }

    } catch (error) {
      console.error(error);
      setAppState('idle');
      alert("Backend API is not running or failed. Ensure FastAPI is running on port 8000.");
    }
  };



  return (
    <main className="main-container">
      <div className="hero-section">
        <span className="badge">Vision LLM + SDXL</span>
        <h1 className="hero-title">Redefine Architecture in <span style={{color: 'var(--accent)'}}>Seconds</span></h1>
        <p className="hero-subtitle">Upload a photo, select a premium 2026 palette, and instantly generate an 8K photorealistic facade renovation.</p>
      </div>

      <div className="workspace">
        {/* Left Sidebar: Controls */}
        <aside className="controls-sidebar">
          <div className="glass-panel controls-card">
            
            {/* Upload Area */}
            <div>
              <div className="section-label" style={{marginBottom: '0.75rem'}}>1. Base Structure</div>
              <label 
                className={`upload-area ${isHovering ? 'active' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setIsHovering(true); }}
                onDragLeave={() => setIsHovering(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsHovering(false);
                  if(e.dataTransfer.files[0]) {
                    setUploadedImage(URL.createObjectURL(e.dataTransfer.files[0]));
                    setAppState('idle');
                  }
                }}
              >
                <div className="upload-icon">📷</div>
                <div className="upload-text">
                  <span>Click to Upload</span> or drag and drop<br/>
                  <small style={{opacity: 0.6}}>High-res JPG / PNG</small>
                </div>
                <input type="file" accept="image/*" style={{display: 'none'}} onChange={handleUpload} />
              </label>
            </div>

            {/* Palette Selection */}
            <div>
              <div className="section-label" style={{marginBottom: '0.75rem'}}>2. Select Aesthetic</div>
              <div className="palette-group">
                {PALETTES.map(palette => (
                  <button 
                    key={palette.id}
                    className={`palette-btn ${selectedPalette === palette.id ? 'selected' : ''}`}
                    onClick={() => setSelectedPalette(palette.id)}
                  >
                    <div className="palette-swatch" style={{background: palette.gradient}}></div>
                    <div className="palette-info">
                      <span className="palette-name">{palette.name}</span>
                      <span className="palette-desc">{palette.desc}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Action Button */}
            <button 
              className="generate-btn"
              disabled={!uploadedImage || appState === 'analyzing' || appState === 'generating'}
              onClick={handleGenerate}
            >
              {appState === 'idle' && (uploadedImage ? "Redesign Facade ✨" : "Upload an Image First")}
              {appState === 'analyzing' && "Analyzing Structure..."}
              {appState === 'generating' && "Rendering 8k Details..."}
              {appState === 'done' && "Regenerate Design"}
            </button>
          </div>
        </aside>

        {/* Right Area: Viewer */}
        <section className="glass-panel viewer-card">
          {!uploadedImage ? (
            <div className="viewer-placeholder">
              <div style={{fontSize: '3rem', marginBottom: '1rem'}}>🏠</div>
              <h3>Your Project Canvas</h3>
              <p style={{maxWidth: '300px', margin: '0 auto'}}>Upload a home photo to the left to begin the AI visualization process.</p>
            </div>
          ) : (
            <div style={{position: 'relative', width: '100%', height: '100%'}}>
              {/* Overlay states */}
              {(appState === 'analyzing' || appState === 'generating') && (
                <div className="loading-overlay" style={{borderRadius: '24px'}}>
                   <div className="spinner"></div>
                   <h3>{appState === 'analyzing' ? 'Analyzing Structure...' : 'Rendering with SDXL...'}</h3>
                   <p style={{color: 'var(--text-secondary)', marginTop: '0.5rem'}}>
                    {appState === 'analyzing' ? 'GPT-4o is reading the facade masks.' : 'Replicate SDXL is painting your design.'}
                   </p>
                </div>
              )}

              {/* Main image: swap between original and generated */}
              <img
                src={(appState === 'done' && generatedImage && showGenerated) ? generatedImage : uploadedImage!}
                alt="Facade"
                style={{
                  width: '100%', height: '100%', objectFit: 'cover',
                  borderRadius: '24px',
                  filter: (appState === 'analyzing' || appState === 'generating') ? 'blur(4px)' : 'none',
                  transition: 'opacity 0.4s ease'
                }}
              />

              {/* Label pill */}
              {appState === 'done' && generatedImage && (
                <div style={{position: 'absolute', top: 16, left: 16, display: 'flex', gap: '8px', zIndex: 10}}>
                  <button
                    onClick={() => setShowGenerated(false)}
                    style={{
                      padding: '0.4rem 1rem', borderRadius: '20px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem',
                      background: !showGenerated ? 'white' : 'rgba(255,255,255,0.2)',
                      color: !showGenerated ? 'black' : 'white',
                      transition: 'all 0.2s'
                    }}
                  >Original</button>
                  <button
                    onClick={() => setShowGenerated(true)}
                    style={{
                      padding: '0.4rem 1rem', borderRadius: '20px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem',
                      background: showGenerated ? 'var(--accent)' : 'rgba(255,255,255,0.2)',
                      color: 'white',
                      transition: 'all 0.2s'
                    }}
                  >✨ Generated</button>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
