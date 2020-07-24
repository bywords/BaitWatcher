// div id : spiLayer 위에


$(document).ready(function(e) {
    console.log('reading')
    //let button_html = '<div class="button_cont" align="center"><a id="report_clickbait" class="report_button" target="_blank" rel="nofollow noopener">낚시성 기사입니다</a><a id="report_not_clickbait" class="report_button" target="_blank" rel="nofollow noopener">낚시성 기사가 아닙니다</a></div>'
    let button_html = '<div class="button_cont" align="center"><a id="report_clickbait" class="report_button" target="_blank" rel="nofollow noopener">Report Clickbait</a><a id="report_not_clickbait" class="report_button" target="_blank" rel="nofollow noopener">Report Not Clickbait</a></div>'
    $(button_html).insertBefore( "#spiLayer" );
    let is_reported = false;
    $("#report_clickbait").click( function(e) {
        if (is_reported) {
            alert("피드백은 1회만 가능합니다.")
        } else {
            if (confirm("피드백을 보내시겠습니까? (낚시성 기사이다)")) {
                alert("요청이 처리되었습니다.")
                is_reported = true;
            }
        }
    });
    $("#report_not_clickbait").click( function(e) {
        if (is_reported) {
            alert("피드백은 1회만 가능합니다.")
        } else {
            if (confirm("피드백을 보내시겠습니까? (낚시성 기사가 아니다)")) {
                alert("요청이 처리되었습니다.")
                is_reported = true;
            }
        }
    })

    attach_popup(article_regex);
});

