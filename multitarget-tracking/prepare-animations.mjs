import fs from 'node:fs';

// Change GIF timing metadata only. Palette, compressed pixels, frame ordering,
// transparency, and disposal methods remain byte-for-byte intact.
export function inspectGif(bytes) {
  if (!['GIF87a', 'GIF89a'].includes(bytes.toString('ascii', 0, 6))) throw new Error('Invalid GIF header');
  const globalTableSize = bytes[10] & 128 ? 3 * 2 ** ((bytes[10] & 7) + 1) : 0;
  const dataStart = 13 + globalTableSize;
  let pos = dataStart;
  const delays = [], loops = [], images = [];
  function subblocks() {
    while (pos < bytes.length && bytes[pos]) pos += 1 + bytes[pos];
    if (pos >= bytes.length) throw new Error('Truncated GIF subblocks');
    pos++;
  }
  while (pos < bytes.length) {
    const start = pos, type = bytes[pos++];
    if (type === 0x3b) break;
    if (type === 0x21) {
      const label = bytes[pos++];
      if (label === 0xf9) {
        if (bytes[pos] !== 4) throw new Error('Invalid graphic control extension');
        delays.push({offset:pos + 2, value:bytes.readUInt16LE(pos + 2)});
        subblocks();
      } else if (label === 0xff) {
        const size = bytes[pos++], app = bytes.toString('ascii', pos, pos + size);
        pos += size;
        if (['NETSCAPE2.0', 'ANIMEXTS1.0'].includes(app) && bytes[pos] === 3 && bytes[pos + 1] === 1) {
          loops.push({offset:pos + 2, value:bytes.readUInt16LE(pos + 2)});
        }
        subblocks();
      } else subblocks();
    } else if (type === 0x2c) {
      const packed = bytes[pos + 8];
      pos += 9;
      if (packed & 128) pos += 3 * 2 ** ((packed & 7) + 1);
      pos++; // LZW minimum code size.
      subblocks();
      images.push({start, end:pos});
    } else throw new Error(`Unexpected GIF block ${type} at ${start}`);
  }
  return {width:bytes.readUInt16LE(6), height:bytes.readUInt16LE(8), dataStart, delays, loops, images};
}

export function fasterLoopingGif(original, speed = 2) {
  const info = inspectGif(original), output = Buffer.from(original);
  if (info.images.length !== info.delays.length) throw new Error('Every frame must have explicit timing');
  let elapsed = 0, written = 0;
  for (const {offset, value} of info.delays) {
    elapsed += value / speed;
    // Accumulate rounding so 25-centisecond frames alternate between 12 and 13.
    const delay = Math.max(2, Math.round(elapsed) - written);
    output.writeUInt16LE(delay, offset);
    written += delay;
  }
  for (const {offset} of info.loops) output.writeUInt16LE(0, offset); // 0 = infinite.
  if (info.loops.length) return output;
  const loop = Buffer.from([0x21,0xff,0x0b,...Buffer.from('NETSCAPE2.0'),0x03,0x01,0x00,0x00,0x00]);
  return Buffer.concat([output.subarray(0,info.dataStart),loop,output.subarray(info.dataStart)]);
}

export function prepareAnimations() {
  const assets = new URL('./assets/', import.meta.url), destination = new URL('playback/', assets);
  fs.mkdirSync(destination, {recursive:true});
  for (const name of fs.readdirSync(assets).filter(name => name.endsWith('.gif'))) {
    fs.writeFileSync(new URL(name.replace('.gif','-2x.gif'), destination), fasterLoopingGif(fs.readFileSync(new URL(name, assets))));
  }
}
