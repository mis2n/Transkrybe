// initializing global variables
var rects = [];

const oplow = 0.2;
const ophigh = 100;
const canvasPad = 10;
const strokeW = 3;
var toggle = 0;
var colorSpace = ['#3783FF', '#4DE94C', '#FF8C00', '#FFEE00', '#F60000'];

// create and display canvas with image as background
var img = document.createElement('img');
img.src = '/static/data/current' + '.jpg' //imgext;
let canvh = img.height;
let canvw = img.width;

var imgInstance = new fabric.Image(img, {
    lockMovementX: true,
    lockMovementY: true,
    width: canvw,
    height: canvh,
    left: canvasPad/2,
    top: canvasPad/2,
});

const canvas = new fabric.Canvas('canvas', {
    backgroundImage: imgInstance,
    width: canvw+canvasPad,
    height: canvh+canvasPad,
    backgroundcolor: 'grey',
    left: '100%',
});
canvas.requestRenderAll();

// Function to dynamically create circles
function makeCircs(cx, cy, w, h, c, idn) {
    let cid = 'c' + idn;
    if (w < h) {
        var crad = w / 2;
    }
    else {
        var crad = h / 2;
    }
    var drad = crad / 40;
    var circle = new fabric.Circle({
        radius: crad, //(w + h) / 4, //1/2 the average of width and height
        fill: '',
        stroke: c,
        left: cx - crad,
        top: cy - crad,
        strokeWidth: 3
    });

    var dot = new fabric.Circle({
        radius: crad / 40, //5% of 1/2 the average of width and height
        fill: c,
        stroke: c,
        left: cx - drad,
        top: cy - drad,
        strokeWidth: 3
    });

    var group = new fabric.Group([circle, dot], {
        selectable: true,
        hasControls: false,
        lockMovementX: true,
        lockMovementY: true,
        lockRotation: true, // disable rotation
        id: cid

    });
    group.setControlsVisibility({ mtr: false}); // hide rotation control
    canvas.add(group);
    canvas.requestRenderAll();
}

// import and parse data from json file (data includes character locations, line associations, and classification data)
import datafile from './data/locs_lines_chars.json' assert { type: "json"};
let data = Object.values(datafile);
let l =data.length;

// Get number of lines on fragment
var linenums = [];
var linecolors = [];
for (let i=0; i<l; i++) {
    linenums.push(data[i].line)
}
var lineset = new Set(linenums)
let linearr = Array.from(lineset);
var colorct = linearr.length;

// setup colorspace for number of lines in fragment
var ctog = 0;
var rcolor;
for (let i=0; i<colorct; i++) {
    linecolors.push(colorSpace[ctog]);
    ctog++;
    if (ctog >= colorSpace.length) {
        ctog = 0;
    }
}

// Build circles and add to canvas
for (let i=0; i<l; i++) {
    rcolor = linecolors[data[i].line];
    makeCircs(data[i].xc, data[i].yc, data[i].width, data[i].height, rcolor, i);
}

window.onload=function(){
    canvas.requestRenderAll();
  };