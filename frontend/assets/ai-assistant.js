(function(){
  const BASE_URL = "http://127.0.0.1:5000";

  function createEl(tag, className, text){
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text) el.textContent = text;
    return el;
  }

  function renderUI(){
    const toggle = createEl('button', 'ai-assistant-toggle');
    toggle.setAttribute('aria-label', 'Open AI Assistant');
    toggle.innerHTML = '🤖';

    const panel = createEl('div', 'ai-assistant-panel');

    const header = createEl('div', 'ai-assistant-header');
    const title = createEl('h4', '', 'Product Assistant');
    const closeBtn = createEl('button');
    closeBtn.textContent = '✕';
    closeBtn.style.border = 'none';
    closeBtn.style.background = 'transparent';
    closeBtn.style.cursor = 'pointer';
    header.appendChild(title);
    header.appendChild(closeBtn);

    const body = createEl('div', 'ai-assistant-body');

    const footer = createEl('div', 'ai-assistant-footer');
    const input = createEl('input', 'ai-input');
    input.placeholder = 'Ask about this product…';
    const sendBtn = createEl('button', 'ai-send-btn', 'Send');
    footer.appendChild(input);
    footer.appendChild(sendBtn);

    panel.appendChild(header);
    panel.appendChild(body);
    panel.appendChild(footer);

    document.body.appendChild(toggle);
    document.body.appendChild(panel);

    function showPanel(){ panel.style.display = 'flex'; }
    function hidePanel(){ panel.style.display = 'none'; }

    toggle.addEventListener('click', showPanel);
    closeBtn.addEventListener('click', hidePanel);

    // Simple state
    let history = [];
    let currentContext = { product_info: null, reviews: null, sales: null };

    function addMessage(role, content){
      const msg = createEl('div', 'ai-msg ' + (role === 'user' ? 'user' : 'assistant'));
      msg.textContent = content;
      body.appendChild(msg);
      body.scrollTop = body.scrollHeight;
    }

    async function fetchSales(productInfo){
      try{
        const res = await fetch(`${BASE_URL}/api/sales-summary`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ category: productInfo && productInfo.source ? productInfo.source : '' , product_title: productInfo && productInfo.title ? productInfo.title : '' })
        });
        if(!res.ok) return null;
        const data = await res.json();
        return data;
      }catch(err){
        return null;
      }
    }

    async function ensureContextFromVisibleProduct(){
      // Try to infer product context from DOM in index page product section if present.
      // Otherwise, leave null; users can paste a URL in chat and we'll fetch.
      const productDisplay = document.getElementById('productDisplay');
      if (!productDisplay) return;
      const img = productDisplay.querySelector('img');
      const titleEl = productDisplay.querySelector('h4, h3, .product-title');
      const priceEl = productDisplay.querySelector('.price, .product-price');
      if (titleEl){
        currentContext.product_info = {
          title: titleEl.textContent.trim(),
          image: img ? img.src : '',
          price: priceEl ? priceEl.textContent.trim() : 'N/A',
          rating: 'N/A',
          source: 'Unknown'
        };
        currentContext.sales = await fetchSales(currentContext.product_info);
      }
    }

    function isLikelyUrl(text){
      return /^(https?:\/\/)/i.test(text) && (text.includes('amazon.') || text.includes('flipkart.'));
    }

    async function fetchProductByUrl(url){
      const res = await fetch(`${BASE_URL}/api/fetch_product_details`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product_url: url })
      });
      if(!res.ok){
        const e = await res.json().catch(()=>({error:'Failed'}));
        throw new Error(e.error || 'Failed to fetch product');
      }
      const data = await res.json();
      return data;
    }

    async function sendMessage(){
      const text = (input.value || '').trim();
      if(!text) return;

      addMessage('user', text);
      history.push({ role: 'user', content: text });
      input.value = '';

      try {
        // If message contains a URL, fetch product details first
        if(isLikelyUrl(text)){
          addMessage('assistant', 'Fetching product details…');
          const details = await fetchProductByUrl(text);
          currentContext.product_info = details.product_info || null;
          currentContext.reviews = details.reviews || [];
          currentContext.sales = await fetchSales(currentContext.product_info);
        } else if (!currentContext.product_info) {
          await ensureContextFromVisibleProduct();
        }

        const res = await fetch(`${BASE_URL}/api/ai/assist`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            history: history,
            product_info: currentContext.product_info,
            reviews: currentContext.reviews
          })
        });
        const data = await res.json();
        const reply = data.reply || 'I could not generate a response right now.';
        addMessage('assistant', reply);
        history.push({ role: 'assistant', content: reply });
      } catch (err){
        addMessage('assistant', 'Something went wrong. You can paste an Amazon/Flipkart URL to analyze it.');
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keydown', function(e){ if(e.key === 'Enter'){ sendMessage(); }});

    // Initial assistant prompt
    addMessage('assistant', 'Hi! Paste an Amazon/Flipkart URL or tell me your needs. I will summarize the product, explain what it is used for, ask your requirements, and suggest if it fits you with recent sales context.');
  }

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', renderUI);
  } else {
    renderUI();
  }
})();