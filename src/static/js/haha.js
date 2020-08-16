

window.onload = function(){

    var img = document.getElementById("user_image");
    img.onchange = function () {
        console.log("123");
    }

    var audio = document.getElementById("user_audio");
    audio.onchange = function () {
        console.log("audio");
    }
}
