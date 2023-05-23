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
});

canvas.requestRenderAll();

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
    //console.log(dex)
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
var sw = screen.width / 2;

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

// create rectangles for each character
for (let i=0; i<l; i++) {
    if (yconf[i] >= 0.75) {var rcolor = colorSpace[1];}
    else if (yconf[i] >= 0.5) {var rcolor = colorSpace[2];}
    else if (yconf[i] >= 0.25) {var rcolor = colorSpace[3];}
    else {var rcolor = colorSpace[4];}
    makeRect(xc[i]-width[i]/2, yc[i]-height[i]/2, width[i], height[i], rcolor, i);
}

// Initialize buttons for listeners
var b1 = document.getElementById("b1");
var b2 = document.getElementById("b2");
var b3 = document.getElementById("b3");
var b4 = document.getElementById("b4");
var b5 = document.getElementById("b5");
var b6 = document.getElementById("b6");
var b7 = document.getElementById("b7");

// Listener for button 1: Update
b1.addEventListener('click', () => {
    // for (let i=0; i<rects.length; i++) {
    //     var _left = rects[i].left;
    //     var _top = rects[i].top;
    //     var _width = rects[i].getScaledWidth()-rects[i].strokeWidth;
    //     var _height = rects[i].getScaledHeight()-rects[i].strokeWidth;
    //     var cbox = [_left, _top, _width, _height];
    //     console.log(cbox);
        
    // }
    console.log("*******************************************************************")
});

// Listener for button 2: Show all
b2.addEventListener('click', () => {
    let j = 0;
    while (j < rects.length) {
        rects[j].set('opacity', ophigh);
        rects[j].set('strokeWidth', strokeW);
        j++;
    }
    // toggle = 0;
    canvas.requestRenderAll();
});

// Listener for button 3: Create new box
b3.addEventListener('click', () => {
    let i = rects.length;
    makeRect(width[0], height[0], width[0], height[0], colorSpace[0], i);
});

// Listener for button 4: Delete active box
b4.addEventListener('click', () => {
    let currobj = canvas.getActiveObject();
    if (currobj == null) {
        alert("Nothing selected");
    }
    else {
        if (rects.length == 1) {
            rects = [];
        }
        else{
        let currid = parseInt(currobj['id'][1]);
        canvas.remove(currobj);
        rects.splice(currid, 1);
        }
    }
    // toggle = 0;
    console.log("Removed: ", currobj['id']);
    document.getElementById("b2").click();
});

// Listener for button 5: Print rects array to console
b5.addEventListener('click', () => {
    console.log(rects, "Array Length:", rects.length);
    console.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
});

// Listener for button 6: Toggle through boxes: next box
b6.addEventListener('click', () => {
    console.log(toggle);
    canvas.setActiveObject(rects[toggle]);
    for (let t=0; t<rects.length; t++) {
        if (t != toggle) {
            rects[t].set('opacity', oplow);
        }
    }
    rects[toggle].set('opacity', ophigh);
    canvas.requestRenderAll();
    if (toggle < rects.length-1) {
    toggle++;
    }
    else {
        toggle = 0;
    }
});

// Listener for button 6: Toggle through boxes: previous box
b7.addEventListener('click', () => {
    if (toggle > 0) {
        toggle = toggle - 1;
    }
    else {
        toggle = rects.length-1;
    }
    console.log(toggle);
    for (let t=0; t<rects.length; t++) {
        if (t != toggle) {
            rects[t].set('opacity', oplow);
        }
    }
    canvas.setActiveObject(rects[toggle]);
    rects[toggle].set('opacity', ophigh);
    canvas.requestRenderAll();
    
});

window.onload=function(){
    canvas.requestRenderAll();
  };