// Meridian single-page client: streams /v1/ask over SSE and renders cited answers.
(function () {
  "use strict";

  const SESSION_KEY = "meridian.session";
  const MARKER_PATTERN = /\[(c|w):([^\]\s]+)\]/g;
  const INSUFFICIENT_STATUS = "insufficient";

  const form = document.getElementById("ask-form");
  const questionInput = document.getElementById("question");
  const askButton = document.getElementById("ask-button");
  const statusLine = document.getElementById("status");
  const card = document.getElementById("answer-card");
  const banner = document.getElementById("state-banner");
  const answerBox = document.getElementById("answer");
  const sourcesHeading = document.getElementById("sources-heading");
  const sourcesList = document.getElementById("sources");

  function readSession() {
    try {
      return window.sessionStorage.getItem(SESSION_KEY);
    } catch (error) {
      return null;
    }
  }

  function writeSession(value) {
    try {
      if (value) {
        window.sessionStorage.setItem(SESSION_KEY, value);
      } else {
        window.sessionStorage.removeItem(SESSION_KEY);
      }
    } catch (error) {
      // storage can be unavailable in private windows; the session then lasts one page view
    }
  }

  let sessionId = readSession();

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatDate(value) {
    if (!value) return "unknown";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toISOString().slice(0, 10);
  }

  async function loadIndexInfo() {
    try {
      const response = await fetch("/v1/index");
      if (!response.ok) throw new Error(String(response.status));
      const info = await response.json();
      document.getElementById("index-version").textContent =
        "Index: v" + (info.index_version ?? "?") + (info.snapshot ? " (" + info.snapshot + ")" : "");
      document.getElementById("paper-count").textContent = info.papers + " papers";
      document.getElementById("last-ingest").textContent = "Last ingest: " + formatDate(info.last_ingest);
    } catch (error) {
      document.getElementById("index-version").textContent = "Index: unavailable";
    }
  }

  function citationNumbering(citations) {
    const numberByMarker = new Map();
    citations.forEach(function (citation, index) {
      numberByMarker.set(citation.marker, index + 1);
    });
    return numberByMarker;
  }

  function renderAnswer(text, citations) {
    answerBox.textContent = "";
    const numberByMarker = citationNumbering(citations);
    let cursor = 0;
    for (const match of text.matchAll(MARKER_PATTERN)) {
      answerBox.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      const marker = match[1] + ":" + match[2];
      const number = numberByMarker.get(marker);
      if (number !== undefined) {
        const citation = citations[number - 1];
        const chip = element("a", "chip" + (citation.external ? " external" : ""), String(number));
        chip.href = citation.url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
        chip.title = (citation.external ? "External web source: " : "") + citation.title +
          (citation.section_title ? ", section " + citation.section_title : "");
        answerBox.appendChild(chip);
      }
      cursor = match.index + match[0].length;
    }
    answerBox.appendChild(document.createTextNode(text.slice(cursor)));
  }

  function renderSources(citations) {
    sourcesList.textContent = "";
    sourcesHeading.hidden = citations.length === 0;
    citations.forEach(function (citation) {
      const item = element("li");
      const link = element("a", "", citation.title || citation.url);
      link.href = citation.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      item.appendChild(link);
      if (citation.external) {
        item.appendChild(element("span", "badge", "external source"));
      } else if (citation.section_title) {
        item.appendChild(element("span", "section", " · " + citation.section_title));
      }
      if (citation.excerpt) item.appendChild(element("span", "excerpt", citation.excerpt));
      sourcesList.appendChild(item);
    });
  }

  function showBanner(text) {
    banner.textContent = text;
    banner.hidden = !text;
  }

  function renderFinal(payload) {
    answerBox.classList.remove("streaming");
    const citations = payload.citations || [];
    renderAnswer(payload.answer || "", citations);
    renderSources(citations);
    if (payload.status === INSUFFICIENT_STATUS) {
      showBanner("Not enough support in the corpus to answer this question.");
    } else if (payload.fallback === "web") {
      showBanner("The corpus had too little support; sources marked external come from the web.");
    } else {
      showBanner("");
    }
    statusLine.textContent = payload.cached ? "Answered from cache." : "";
  }

  function parseFrames(buffer, onEvent) {
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let eventName = "message";
      let data = "";
      frame.split("\n").forEach(function (line) {
        if (line.startsWith("event: ")) eventName = line.slice(7);
        if (line.startsWith("data: ")) data += line.slice(6);
      });
      if (data) onEvent(eventName, JSON.parse(data));
      boundary = buffer.indexOf("\n\n");
    }
    return buffer;
  }

  async function ask(question) {
    card.hidden = false;
    showBanner("");
    answerBox.textContent = "";
    answerBox.classList.add("streaming");
    sourcesList.textContent = "";
    sourcesHeading.hidden = true;
    statusLine.textContent = "Searching the corpus...";
    askButton.disabled = true;

    const body = { question: question };
    if (sessionId) body.session_id = sessionId;
    try {
      const response = await fetch("/v1/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (response.status === 429) {
        const retry = response.headers.get("Retry-After");
        statusLine.textContent = "Rate limit reached; try again in " + (retry || "a few") + " seconds.";
        answerBox.classList.remove("streaming");
        return;
      }
      if (!response.ok || !response.body) {
        statusLine.textContent = "The request failed (" + response.status + ").";
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer = parseFrames(buffer + decoder.decode(chunk.value, { stream: true }), function (name, data) {
          if (name === "meta") {
            sessionId = data.session_id;
            writeSession(sessionId);
          } else if (name === "token") {
            statusLine.textContent = "Writing the answer...";
            answerBox.textContent += data.text;
          } else if (name === "reset") {
            answerBox.textContent = "";
            statusLine.textContent = "Rewriting unsupported sentences...";
          } else if (name === "final") {
            renderFinal(data);
          } else if (name === "error") {
            statusLine.textContent = data.detail || "The answer could not be generated.";
          }
        });
      }
    } catch (error) {
      statusLine.textContent = "The connection failed.";
    } finally {
      askButton.disabled = false;
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (question) ask(question);
  });

  questionInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) form.requestSubmit();
  });

  document.getElementById("new-session").addEventListener("click", function () {
    sessionId = null;
    writeSession(null);
    card.hidden = true;
    questionInput.value = "";
    questionInput.focus();
  });

  loadIndexInfo();
})();
