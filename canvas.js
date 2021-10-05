import {forward} from "./forwarding.js"

var canvas = document.querySelector("canvas");
var cobj = canvas.getContext("2d");
var shape = document.querySelector("#xing");
var redo = document.querySelector("#redo");
var qingkong = document.querySelector("#qingkong");
var output = document.querySelector("h1")

canvas.style.height = 500
canvas.style.width = 500
canvas.height = 28
canvas.width = 28

var data = [];
var s = "pen";
shape.onchange = function() {
    s = this.value;
};
var c = "#000";
var w = "4";
var st = "stroke";

canvas.onmousedown = function(e) {
    var ox = e.offsetX*canvas.width/canvas.style.width;
    var oy = e.offsetY*canvas.height/canvas.style.height;
    var draw = new Draw(cobj, {
        color: c,
        width: w,
        style: st
    });
    if (s == "pen") {
        cobj.beginPath();
        cobj.moveTo(ox, oy);
    }
    canvas.onmousemove = function(e) {
        var mx = e.offsetX*canvas.width*1.0/parseInt(canvas.style.width);
        var my = e.offsetY*canvas.height*1.0/parseInt(canvas.style.height);
        if (s != "eraser") {
            cobj.clearRect(0, 0, canvas.width,canvas.height);
            if (data.length != 0) {
                cobj.putImageData(data[data.length - 1], 0, 0, 0, 0, canvas.width,canvas.height); //将某个图像数据放置到画布指定的位置上  后面四个参数可省略
            }
        }
        draw[s](ox, oy, mx, my);
    };
    document.onmouseup = function() {
        data.push(cobj.getImageData(0, 0, canvas.width,canvas.height)); //获取画布当中指定区域当中所有的图形数据
        canvas.onmousemove = null;
        document.onmouseup = null;
        let picture = [new Array()];
        for(let i=0;i<28;i++){
            picture[0][i]=new Array();
            for(let j=0;j<28;j++)
                picture[0][i][j]=0;
        }
        cobj.getImageData(0, 0, canvas.width,canvas.height).data.map((value,index)=>{
            if((index-3)%4===0){
                picture[0][Math.floor(((index-3)/4)/28.0)][Math.round(((index-3)/4)%28)] = value
            }
            return value;
        })
        let out = forward(picture)
        output.innerText = "I think it is: "+out.toString()
    }
};
redo.onclick = function() {
    if (data.length == 0) {
        return;
    }
    cobj.clearRect(0, 0, canvas.width,canvas.height);
    data.pop();
    if (data.length == 0) {
        return;
    }
    cobj.putImageData(data[data.length - 1], 0, 0, 0, 0, canvas.width,canvas.height);
};
qingkong.onclick = function() {
    cobj.clearRect(0, 0, canvas.width,canvas.height);
    data = [];
}
class Draw {
    constructor(cobj, option) {
        this.cobj = cobj;
        this.color = option.color;
        this.width = option.width;
        this.style = option.style;
    }
    init() { //初始化
        this.cobj.strokeStyle = this.color;
        this.cobj.fillStyle = this.color;
        this.cobj.lineWidth = this.width;
    }
    pen(ox, oy, mx, my) {
        this.init();
        this.cobj.lineTo(mx, my);
        this.cobj.stroke();
    }
    eraser(ox, oy, mx, my) {

        this.cobj.clearRect(mx, my, 10, 10);
    }

}