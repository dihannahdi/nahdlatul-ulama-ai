"""
Progressive Document Loader
Allows you to choose how many documents to process: 500, 2000, 5000, or ALL
"""

import os
import asyncio
from pathlib import Path

class ProgressiveLoader:
    """Load documents progressively based on user choice"""
    
    def __init__(self):
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        self.sql_chunks_dir = os.path.join(project_root, "sql_chunks")
    
    def get_document_counts(self):
        """Get available document processing options"""
        sql_files = [f for f in os.listdir(self.sql_chunks_dir) if f.endswith('.sql')]
        total_files = len(sql_files)
        
        options = {
            "fast": 500,
            "medium": 2000, 
            "large": 5000,
            "complete": total_files
        }
        
        print("📊 Document Processing Options:")
        print(f"  🚀 Fast Mode: {options['fast']} documents (~10 seconds)")
        print(f"  ⚡ Medium Mode: {options['medium']} documents (~30 seconds)")
        print(f"  🔥 Large Mode: {options['large']} documents (~90 seconds)")
        print(f"  🏭 Complete Mode: {options['complete']} documents (~5-10 minutes)")
        
        return options

def create_configurable_backend():
    """Create a backend with configurable document count"""
    
    loader = ProgressiveLoader()
    options = loader.get_document_counts()
    
    print("\n🎯 Choose your processing mode:")
    print("1. Fast (500 docs) - Development & testing")
    print("2. Medium (2000 docs) - Balanced coverage") 
    print("3. Large (5000 docs) - High coverage")
    print("4. Complete (ALL docs) - Maximum coverage")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                mode = "fast"
                max_docs = options["fast"]
                break
            elif choice == "2":
                mode = "medium"
                max_docs = options["medium"]
                break
            elif choice == "3":
                mode = "large"
                max_docs = options["large"]
                break
            elif choice == "4":
                mode = "complete"
                max_docs = options["complete"]
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                continue
                
        except KeyboardInterrupt:
            print("\nDefaulting to fast mode...")
            mode = "fast"
            max_docs = options["fast"]
            break
    
    print(f"\n✅ Selected: {mode.title()} Mode ({max_docs} documents)")
    
    # Update the ultra_fast_main.py processor
    update_processor_config(max_docs, mode)
    
    return mode, max_docs

def update_processor_config(max_docs: int, mode: str):
    """Update the processor configuration"""
    
    # Read the current ultra_fast_main.py
    with open("ultra_fast_main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update the sample size in UltraFastProcessor
    if "sample_size = min(500, total_files)" in content:
        new_line = f"sample_size = min({max_docs}, total_files)  # {mode} mode"
        content = content.replace("sample_size = min(500, total_files)", new_line)
    
    # Update the max_docs in UltraFastProcessor initialization
    if "UltraFastProcessor(max_docs: int = 1000)" in content:
        new_line = f"UltraFastProcessor(max_docs: int = {max_docs})"
        content = content.replace("UltraFastProcessor(max_docs: int = 1000)", new_line)
    
    # Update the processor initialization
    if "processor = UltraFastProcessor()" in content:
        new_line = f"processor = UltraFastProcessor({max_docs})"
        content = content.replace("processor = UltraFastProcessor()", new_line)
    
    # Write back the updated content
    with open(f"configurable_main_{mode}.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"📝 Created configurable_main_{mode}.py with {max_docs} document limit")

if __name__ == "__main__":
    mode, max_docs = create_configurable_backend()
    
    print(f"\n🚀 To run your configured backend:")
    print(f"python configurable_main_{mode}.py")
    
    print(f"\n📊 Expected performance:")
    if mode == "fast":
        print("  • Startup: 5-10 seconds")
        print("  • Memory: ~100MB") 
        print("  • Coverage: Good for development")
    elif mode == "medium":
        print("  • Startup: 20-40 seconds")
        print("  • Memory: ~300MB")
        print("  • Coverage: Balanced production")
    elif mode == "large": 
        print("  • Startup: 60-120 seconds")
        print("  • Memory: ~800MB")
        print("  • Coverage: High production")
    else:  # complete
        print("  • Startup: 5-15 minutes (first time)")
        print("  • Memory: ~2GB")
        print("  • Coverage: Complete database")
        print("  • Note: Use full_production_main.py instead")
