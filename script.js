const tabButtons = document.querySelectorAll(".about-type");
const tabContents = document.querySelectorAll(".tab-content");

tabButtons.forEach((button, index) => {
  button.addEventListener("click", () => {
    tabButtons.forEach(btn => btn.classList.remove("active-tab"));
    tabContents.forEach(content => content.classList.remove("active-tab"));

    button.classList.add("active-tab");
    tabContents[index].classList.add("active-tab");
  });
});

// === Scroll-based fade-in for Interests and other sections ===
const scrollElements = document.querySelectorAll('.animate-on-scroll');

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target); // Optional: only animate once
    }
  });
}, {
  threshold: 0.2 // 20% visible before triggering
});

scrollElements.forEach(el => observer.observe(el));

// === Optional openTab function (if using elsewhere) ===
function openTab(evt, tabName) {
  const tabContents = document.querySelectorAll(".tab-content");
  const tabButtons = document.querySelectorAll(".tab");

  tabContents.forEach(tab => tab.classList.add("hidden"));
  tabButtons.forEach(btn => btn.classList.remove("active"));

  document.getElementById(tabName).classList.remove("hidden");
  evt.currentTarget.classList.add("active");
}

let slideIndex = 1;
let slideTimer;

function showSlides(n) {
  const slides = document.querySelectorAll(".slide");
  const dots = document.querySelectorAll(".dot");

  if (n > slides.length) slideIndex = 1;
  if (n < 1) slideIndex = slides.length;

  slides.forEach(slide => slide.style.display = "none");
  dots.forEach(dot => dot.classList.remove("active-dot"));

  slides[slideIndex - 1].style.display = "block";
  dots[slideIndex - 1].classList.add("active-dot");

  clearTimeout(slideTimer);
  slideTimer = setTimeout(() => changeSlide(1), 4000); // Auto slide every 4s
}

function changeSlide(n) {
  showSlides(slideIndex += n);
}

function setSlide(n) {
  showSlides(slideIndex = n);
}

// Initialize slideshow
document.addEventListener("DOMContentLoaded", () => {
  showSlides(slideIndex);
});

function openmenu() {
  document.getElementById("sidemenu").classList.add("show");
}

function closemenu() {
  document.getElementById("sidemenu").classList.remove("show");
}

function toggleMenu() {
  const menu = document.getElementById("sidemenu");
  const icon = document.querySelector(".menu-toggle");

  menu.classList.toggle("show");

  if (menu.classList.contains("show")) {
    icon.classList.remove("fa-bars");
    icon.classList.add("fa-xmark");
  } else {
    icon.classList.remove("fa-xmark");
    icon.classList.add("fa-bars");
  }
}