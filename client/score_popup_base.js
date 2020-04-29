const article_regex = new RegExp("/main/(ranking/)?read.nhn*")
//const article_regex = new RegExp("(/movie/now|/ranking|/now|/starcast)/read*")
//const article_regex = new RegExp("/(movie|ranking|now|starcast)*")


let number_flag = true;

chrome.storage.sync.get(['global_flag'], function(result) {
    number_flag = result.global_flag;
  });

function attach_popup(article_regex, base_url) {
    let target_a_tags = $('a')
        .filter(function(index){
            return article_regex.test($(this).attr('href'))
        })
    target_a_tags.attr('data-tippy-content', "확률을 계산하는 중입니다...");
    target_a_tags.attr('score_target', true);
    target_a_tags.attr('score', -1);
    tippy(target_a_tags.toArray(), {
        arrow: true
    });
    target_a_tags
        .one('mouseover', function(e) {
            let cur = $(this);
            let cur_tippy = cur[0]._tippy;
            let href_url = cur.attr('href')
            if (href_url[0] === "/") {
                href_url = base_url + href_url;
            }
            $.ajax({
                type:"POST",
                //url:"http://localhost:12345/api/articles/",
                url:"https://incongruent-detection-2.tk/api/articles/",
                crossDomain: true,
                data: {
                    'url': href_url
                },
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                dataType: "JSON",
                success: function(json) {
                    console.log('Post Returned!')
                    let score = Math.floor(parseFloat(json.score) * 100)
                    score = Math.floor(score)
                    if (number_flag) {
                        cur_tippy.setContent("기사의 정합성이 낮을 확률 : <b>" + score + "%</b>");
                    } else {
                        let label_text = null;
                        if (score < 50) {
                            label_text = "낮음"; 
                        } else if (score < 75) {
                            label_text = "중간"
                        } else {
                            label_text = "높음"
                        }
                        cur_tippy.setContent("기사의 정합성이 떨어질 확률 <b>" + label_text + "</b>");
                    }
                },
                timeout: 5000,
                error: function (json) {
                    cur_tippy.setContent("점수 계산에 실패하였습니다.");
                }
            });
    });
}

function attach_popup_update(article_regex, base_url) {
    let target_a_tags = $('a')
        .filter(function(index){
            //return article_regex.test($(this).attr('href'))
            return article_regex.test($(this).attr('href'))
        })
        .not('[score=-1]')
    target_a_tags.attr('data-tippy-content', "확률을 계산하는 중입니다...");
    target_a_tags.attr('score_target', true);
    target_a_tags.attr('score', -1);
    tippy(target_a_tags.toArray(), {
        arrow: true
    });
    target_a_tags
        .one('mouseover', function(e) {
            let cur = $(this);
            let cur_tippy = cur[0]._tippy;
            let href_url = cur.attr('href')
            if (href_url[0] === "/") {
                href_url = base_url + href_url;
            }
            $.ajax({
                type:"POST",
                url:"https://incongruent-detection-2.tk/api/articles/",
                data: {
                    'url': href_url
                },
                dataType:"JSON",
                success: function(json) {
                    console.log('Post Returned!')
                    let score = Math.floor(parseFloat(json.score) * 100)
                    score = Math.floor(score)
                    if (number_flag) {
                        cur_tippy.setContent("기사의 정합성이 낮을 확률 : <b>" + score + "%</b>");
                    } else {
                        let label_text = null;
                        if (score < 50) {
                            label_text = "낮음"; 
                        } else if (score < 75) {
                            label_text = "중간"
                        } else {
                            label_text = "높음"
                        }
                        cur_tippy.setContent("기사의 정합성이 떨어질 확률 <b>" + label_text + "</b>");
                    }
                },
                timeout: 5000,
                error: function (json) {
                    cur_tippy.setContent("점수 계산에 실패하였습니다.");
                }
            });
    });
}

$(document).ready(function(e) {
    console.log('naver_version_loaded')
    let base_url = "https://" + window.location.href.split("/")[2];
    window.setTimeout(function (e) {
        console.log("UPDATE Clicked!")
        attach_popup(article_regex, base_url);
    }, 1000);
    attach_popup(article_regex, base_url)
    $("._news_cluster_more_btn").click(function (e) {
        window.setTimeout(function (e) {
            console.log("UPDATE Clicked!")
            attach_popup_update(article_regex, base_url);
        }, 1000);
    })
    $(".nclicks\\(rig\\.ranking\\)").on("mouseover", function (e) {
        window.setTimeout(function (e) {
            console.log("Ranking Update Clicked!")
            attach_popup_update(article_regex, base_url);
        }, 200);
    })
});