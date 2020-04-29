const article_regex = new RegExp("news.naver.com/main/read.nhn*")
//const article_regex = new RegExp("now/read*")
const insight_regex = new RegExp("^http://www.insight.co.kr/news*")
/*
$(document).ready(function(e) {
    $("a").hover(function(e) {
        let link = $(this).attr('href');
        if (article_regex.test(link)) {
            alert($(this).attr('href'));
        }
    });
});*/

window.onload = function() {
    console.log('loaded')
    let a_elems = document.getElementsByTagName('a');
    let poppers = [];
    //a_elems = a_elems.filter(elem => article_regex.test(elem.getAttribute("href")));
    for (let i=0; i < a_elems.length; i++) {
        if (article_regex.test(a_elems[i].getAttribute("href"))) {
            console.log(a_elems[i].innerText);
            //let popper_param = createNewPopup();
            //let popper = new Popper(a_elems[i], popper_param);
            let tooltip = new Tooltip(a_elems[i]);
            console.log(tooltip);
            tooltip.updateTitleContent("TESTETSTETSTSTSTST");
        }
    }
    console.log("Finished");
}

function createNewPopup()
{
    newDiv = document.createElement("div");
    newDiv.className += "popper";
    newDiv.innerHTML = "<p class=\"bold\">Popper on </p>";
    document.body.appendChild(newDiv)
    return newDiv
}