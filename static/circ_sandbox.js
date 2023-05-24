// initializing global variables
var rects = [];

const oplow = 0.2;
const ophigh = 100;
const canvasPad = 10;
const strokeW = 3;
var toggle = 0;

//Temporary data for demo and testing
// *********************************************************************

const chardata = [[83, 76],
                [18, 34, 26, 86, 31, 8, 5],
                [9, 66, 89, 21, 46, 52, 1, 7, 42, 6, 33, 80, 84, 78],
                [68, 55, 81, 96, 28, 54, 17, 56, 44, 57, 58, 29, 19, 79],
                [49, 3, 61, 30, 4, 59, 10, 47, 60, 15],
                [12, 0, 32, 43, 69, 62, 53, 20, 24, 74, 45],
                [22, 14, 72, 88, 11, 71, 37, 97, 73, 51, 36],
                [63, 2, 39, 35, 82, 48, 41, 65, 16, 64, 77, 50, 38, 100],
                [98, 25, 70, 23, 27, 13, 40, 67]];

// *********************************************************************

var colorSpace = ['#3783FF', '#4DE94C', '#FF8C00', '#FFEE00', '#F60000'];
var lineColors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

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
});
canvas.requestRenderAll();

//**********************************************************************************
// var img2 = document.createElement('img2');
// img2.src = '/static/line_output.jpg'

// var imgInstance2 = new fabric.Image(img2, {
//     lockMovementX: true,
//     lockMovementY: true,
//     width: canvw,
//     height: canvh,
//     left: canvasPad/2,
//     top: canvasPad/2,
// });

// const canvas2 = new fabric.Canvas('canvas2', {
//     backgroundImage: imgInstance2,
//     width: canvw+canvasPad,
//     height: canvh+canvasPad,
//     backgroundcolor: 'grey',
// });

// canvas2.requestRenderAll();
//**********************************************************************************

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
        selectable: false,
        lockRotation: true, // disable rotation
        id: cid

    });

    group.setControlsVisibility({ mtr: false}); // hide rotation control
    canvas.add(group);
    canvas.requestRenderAll();
}

// Function to dynamically create rectangles
function makeRect(l, r, w, h, c, idn) {
    let cid = 'r' + idn;
    var rect = new fabric.Rect({
        left: l,
        top: r,
        width: w,
        height: h,
        fill: '',
        opacity: 100,
        stroke: c,
        strokeWidth: strokeW,
        selectable: true,
        lockRotation: true, // disable rotation
        id: cid
    });
    rect.setControlsVisibility({ mtr: false}); // hide rotation control
    rect.on("selected", (element) => {
        let dex = rects.findIndex(obj => {
            return obj.id == element.target.id;
        });
        let j = 0;
        while (j < rects.length) {
            if (j != dex) {
                rects[j].set('opacity', oplow);
            }
            else {
                rects[j].set('opacity', ophigh);
                rects[j].set('strokeWidth', 4);
            }
            j++;
        }
    console.log(dex)
});
rects.push(rect);
canvas.add(rect);
canvas.requestRenderAll();
}

// import data from json file
import datafile from './data/current.json' assert { type: "json"};
let data = Object.values(datafile);

// parse data into arrays
var width = data[1];
var height = data[2];
var xc = data[3];
var yc = data[4];
var yconf = data[0];
// var sw = screen.width / 2;

// count number of characters identified by yolo
function countProperties(obj) {
    var count = 0;

    for(var prop in obj) {
        if(obj.hasOwnProperty(prop))
            ++count;
    }

    return count;
}

let l = countProperties(xc);

// // create circle and dot in centroid (and rectangle if uncommented) for each character
// for (let i=0; i<l; i++) {
//     var rcolor = 'blue';
//     // makeRect(xc[i]-width[i]/2, yc[i]-height[i]/2, width[i], height[i], rcolor, i);
//     makeCircs(xc[i], yc[i], width[i], height[i], 'red', i);
// }

var ctog = 0
for (let i=0; i<chardata.length; i++) {
    let rcolor = lineColors[ctog];
    if (ctog < lineColors.length-1) {
        ctog++;
    }
    else {
        ctog = 0;
    }
    for (let j=0; j<chardata[i].length; j++) {
        let ldex = chardata[i][j];
        makeCircs(xc[ldex], yc[ldex], width[ldex], height[ldex], rcolor, ldex);
    }
}

window.onload=function(){
    canvas.requestRenderAll();
  };