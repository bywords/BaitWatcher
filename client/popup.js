$(document).ready(function() {
    chrome.storage.sync.get(['global_flag'], function(result) {
        $('#score-flag').removeAttr("checked");
        $("#score-flag").prop("checked", result.global_flag);
      });
    $("#score-flag").change(function() {
        chrome.storage.sync.get(['global_flag'], function(result) {
            if (result.global_flag) {
                chrome.storage.sync.set({'global_flag': false}, function() {
                    console.log('Changed to false');
                  });
            } else {
                chrome.storage.sync.set({'global_flag': true}, function() {
                    console.log('Changed to true');
                  });
            }
        });
    })
})