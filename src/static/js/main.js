const onInitCarousel = () => {
  const carousel = document.querySelector('.carousel');
  const slides = document.querySelectorAll('.slide');
  const slideCount = slides.length;
  const slideWidth = 608;

  let currentIndex = 0;

  function updateSlideClasses() {
    slides.forEach((slide, index) => {
      slide.classList.remove('active', 'adjacent');
      if (index === currentIndex) {
        slide.classList.add('active');
      } else if (
        index === (currentIndex + 1) % slideCount ||
        index === (currentIndex - 1 + slideCount) % slideCount
      ) {
        slide.classList.add('adjacent');
      }
    });
  }

  // Move carousel
  function moveCarousel() {
    currentIndex = (currentIndex + 1) % slideCount;
    const offset = -currentIndex * slideWidth;
    carousel.style.transform = `translateX(${offset}px)`;
    updateSlideClasses();
  }

  // Initialize carousel
  updateSlideClasses();
  setInterval(moveCarousel, 3000);

}

document.addEventListener('DOMContentLoaded', function() {
  onInitCarousel();
  console.log('JavaScript is loaded and ready!');
});
