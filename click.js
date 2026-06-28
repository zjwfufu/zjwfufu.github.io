! function(e, t) {
    var d = [];
    var l = false;

    function r() {
        for (var i = 0; i < d.length; i++) {
            var p = d[i];
            p.alpha -= 0.015;
            p.y -= 1.5;
            p.scale += 0.005;
            if (p.alpha <= 0) {
                t.body.removeChild(p.el);
                d.splice(i, 1);
                i--;
            } else {
                p.el.style.cssText = "left:" + p.x + "px;top:" + p.y + "px;opacity:" + p.alpha + ";transform:scale(" + p.scale + "," + p.scale + ");position:fixed;pointer-events:none;z-index:99999;width:35px;height:41px;";
            }
        }
        if (d.length > 0) {
            requestAnimationFrame(r);
        } else {
            l = false;
        }
    }

    function spawn(ev) {
        var img = t.createElement("img");
        img.src = "/img/xiaolongbao_with_tianyi.png";
        img.style.cssText = "position:fixed;pointer-events:none;z-index:99999;width:35px;height:41px;left:" + (ev.clientX - 17) + "px;top:" + (ev.clientY - 20) + "px;";
        d.push({
            el: img,
            x: ev.clientX - 17,
            y: ev.clientY - 20,
            scale: 1,
            alpha: 1
        });
        t.body.appendChild(img);
        if (!l) {
            l = true;
            requestAnimationFrame(r);
        }
    }

    e.requestAnimationFrame = e.requestAnimationFrame || e.webkitRequestAnimationFrame || e.mozRequestAnimationFrame || function(cb) { setTimeout(cb, 1e3 / 60); };
    e.addEventListener('click', spawn);
}(window, document);
