// RallyVision — Minimal scripts

document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Subtle nav background on scroll
    const nav = document.querySelector('.nav');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.style.borderBottomColor = '#e5e2dd';
        } else {
            nav.style.borderBottomColor = '#e5e2dd';
        }
    });

    // Add copy functionality to code blocks
    document.querySelectorAll('.code-block').forEach(block => {
        block.addEventListener('click', async () => {
            const code = block.querySelector('code').textContent;
            try {
                await navigator.clipboard.writeText(code);
                
                // Brief visual feedback
                block.style.outline = '2px solid #1a1a1a';
                block.style.outlineOffset = '2px';
                
                setTimeout(() => {
                    block.style.outline = 'none';
                }, 200);
            } catch (err) {
                // Clipboard API not available
            }
        });
        
        // Indicate clickable
        block.style.cursor = 'pointer';
        block.title = 'Click to copy';
    });
});
