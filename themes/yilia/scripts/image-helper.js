'use strict';

var path = require('path');
var fs = require('fs');

// Quick image dimension reader (reads file header only)
function getImageSize(filePath) {
  try {
    var buf = Buffer.alloc(32);
    var fd = fs.openSync(filePath, 'r');
    fs.readSync(fd, buf, 0, 32, 0);
    fs.closeSync(fd);
    // PNG
    if (buf[0] === 0x89 && buf[1] === 0x50) {
      return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
    }
    // JPEG
    fd = fs.openSync(filePath, 'r');
    var jbuf = Buffer.alloc(65536);
    fs.readSync(fd, jbuf, 0, 65536, 0);
    fs.closeSync(fd);
    for (var i = 0; i < jbuf.length - 10; i++) {
      if (jbuf[i] === 0xFF && (jbuf[i+1] === 0xC0 || jbuf[i+1] === 0xC1 || jbuf[i+1] === 0xC2)) {
        return { w: jbuf.readUInt16BE(i + 7), h: jbuf.readUInt16BE(i + 5) };
      }
    }
  } catch (e) {}
  return null;
}

hexo.extend.helper.register('get_landscape_images', function(post, maxCount) {
  var sourceDir = hexo.source_dir;
  var imgs = [];
  if (!post.content) return imgs;

  var re = /<img[^>]+src=["']([^"']+)["'][^>]*>/g;
  var m;
  while ((m = re.exec(post.content)) !== null && imgs.length < maxCount) {
    var src = m[1];
    // Resolve to local file
    var localPath = src.startsWith('/') ? path.join(sourceDir, src) : null;
    if (localPath && fs.existsSync(localPath)) {
      var dims = getImageSize(localPath);
      if (dims) {
        if (dims.w < 80 || dims.h < 80) continue;
        if (dims.h > dims.w * 1.2) continue;
      }
    }
    imgs.push(src);
  }
  return imgs;
});
