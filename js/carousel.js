/* Lightweight carousel - replaces Bootstrap carousel */
(function () {
    document.querySelectorAll('.carousel.slide').forEach(function (el) {
        var items = el.querySelectorAll('.carousel-item');
        if (items.length === 0) return;

        var current = 0;
        var animating = false;
        items.forEach(function (item, i) {
            if (item.classList.contains('active')) current = i;
        });

        function show(index) {
            if (animating) return;
            var next = (index + items.length) % items.length;
            if (next === current) return;

            animating = true;
            var outItem = items[current];
            var inItem = items[next];

            // make incoming item visible but transparent (positioned absolute)
            inItem.style.position = 'absolute';
            inItem.style.opacity = '0';
            inItem.classList.add('active');

            // force reflow so transition triggers
            inItem.offsetHeight;

            // fade in new, fade out old
            inItem.style.opacity = '1';
            outItem.style.opacity = '0';

            setTimeout(function () {
                outItem.classList.remove('active');
                outItem.style.opacity = '';
                outItem.style.position = '';
                inItem.style.position = '';
                inItem.style.opacity = '';
                current = next;
                animating = false;
            }, 500);
        }

        var prev = el.querySelector('.carousel-control-prev');
        var next = el.querySelector('.carousel-control-next');
        if (prev) prev.addEventListener('click', function (e) { e.preventDefault(); show(current - 1); });
        if (next) next.addEventListener('click', function (e) { e.preventDefault(); show(current + 1); });
    });
})();
