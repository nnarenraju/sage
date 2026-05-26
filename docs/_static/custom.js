/* Sage docs — dynamic enhancements (no style/colour changes) */

(function () {
  "use strict";

  /* ------------------------------------------------------------------
     1. Reading progress bar
     A thin bar fixed at the very top of the viewport that fills as
     the user scrolls through the page.
  ------------------------------------------------------------------ */
  function initProgressBar() {
    var bar = document.createElement("div");
    bar.id = "sage-progress-bar";
    document.body.appendChild(bar);

    function update() {
      var scrollTop = window.scrollY || document.documentElement.scrollTop;
      var docHeight =
        document.documentElement.scrollHeight -
        document.documentElement.clientHeight;
      var pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      bar.style.width = pct + "%";
    }

    window.addEventListener("scroll", update, { passive: true });
    update();
  }

  /* ------------------------------------------------------------------
     2. Scroll-triggered fade-in
     Headings and block-level content fade + slide up into view as
     they enter the viewport. Elements start invisible (set via CSS
     class .sage-reveal) and gain .sage-visible once observed.
  ------------------------------------------------------------------ */
  function initFadeIn() {
    if (!window.IntersectionObserver) return;

    var targets = document.querySelectorAll(
      ".rst-content h2, " +
      ".rst-content h3, " +
      ".rst-content .admonition, " +
      ".rst-content figure, " +
      ".rst-content table.docutils, " +
      ".rst-content .highlight, " +
      ".sd-card"
    );

    targets.forEach(function (el) {
      el.classList.add("sage-reveal");
    });

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("sage-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );

    targets.forEach(function (el) {
      observer.observe(el);
    });
  }

  /* ------------------------------------------------------------------
     3. Back-to-top button
     Appears after the user scrolls 400 px down; smooth-scrolls back.
  ------------------------------------------------------------------ */
  function initBackToTop() {
    var btn = document.createElement("button");
    btn.id = "sage-back-to-top";
    btn.title = "Back to top";
    btn.setAttribute("aria-label", "Scroll back to top");
    btn.innerHTML = "&#8679;"; /* ↑ */
    document.body.appendChild(btn);

    function syncVisibility() {
      var scrolled = window.scrollY || document.documentElement.scrollTop;
      btn.classList.toggle("sage-btt-visible", scrolled > 400);
    }

    window.addEventListener("scroll", syncVisibility, { passive: true });

    btn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    syncVisibility();
  }

  /* ------------------------------------------------------------------
     4. Figure lightbox
     Clicking any inline figure image opens a full-viewport overlay
     so the user can inspect high-resolution diagrams (e.g. the
     methodology flowchart) without leaving the page.
  ------------------------------------------------------------------ */
  function initLightbox() {
    var overlay = document.createElement("div");
    overlay.id = "sage-lightbox";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Image viewer");

    var img = document.createElement("img");
    overlay.appendChild(img);

    var closeBtn = document.createElement("button");
    closeBtn.id = "sage-lightbox-close";
    closeBtn.innerHTML = "&times;";
    closeBtn.setAttribute("aria-label", "Close image viewer");
    overlay.appendChild(closeBtn);

    document.body.appendChild(overlay);

    function open(src, alt) {
      img.src = src;
      img.alt = alt || "";
      overlay.classList.add("sage-lightbox-open");
      document.body.style.overflow = "hidden";
    }

    function close() {
      overlay.classList.remove("sage-lightbox-open");
      document.body.style.overflow = "";
      img.src = "";
    }

    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) close();
    });
    closeBtn.addEventListener("click", close);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });

    document.querySelectorAll(".rst-content figure img").forEach(function (el) {
      if (el.classList.contains("sage-logo-hero")) return;
      el.style.cursor = "zoom-in";
      el.addEventListener("click", function () {
        open(el.src, el.alt);
      });
    });
  }

  /* ------------------------------------------------------------------
     5. Logo download button
     Injected below the hero logo on the index page. Does not appear
     on any other page.
  ------------------------------------------------------------------ */
  function initLogoDownload() {
    var logo = document.querySelector("img.sage-logo-hero");
    if (!logo) return;

    var container = logo.closest("figure") || logo.parentElement;

    var btn = document.createElement("a");
    btn.id = "sage-logo-download";
    btn.href = logo.src;
    btn.download = "sage-logo.png";
    btn.title = "Download logo";
    btn.setAttribute("aria-label", "Download Sage logo");
    btn.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" ' +
      'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>' +
      '<polyline points="7 10 12 15 17 10"/>' +
      '<line x1="12" y1="15" x2="12" y2="3"/>' +
      "</svg>" +
      " Download logo";

    container.insertAdjacentElement("afterend", btn);
  }

  /* ------------------------------------------------------------------
     Boot
  ------------------------------------------------------------------ */
  document.addEventListener("DOMContentLoaded", function () {
    initProgressBar();
    initFadeIn();
    initBackToTop();
    initLightbox();
    initLogoDownload();
  });
})();
