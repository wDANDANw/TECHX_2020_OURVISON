

window.onload = function loadImage(){

    var img = document.getElementById("user image");
    img.src = "../rescources/1.jpg";
    img.onload = function () {
        console.log("123");
    }
}