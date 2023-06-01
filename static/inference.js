// initializing global variables
var rects = [];

const oplow = 0.2;
const ophigh = 100;
const canvasPad = 10;
const strokeW = 3;
var toggle = 0;



// create and display canvas with image as background
var img = document.createElement('img');
img.src = '/static/images/line_output.jpg';

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
});
canvas.requestRenderAll();

window.onload=function(){
    canvas.requestRenderAll();
  };