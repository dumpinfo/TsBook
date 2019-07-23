
<script type="text/javascript">
$("div[id^='o_'").each(bind_option_click);
function bind_option_click(idx, elem) {
    $("#" + elem.id).bind("click", optnOnClick);
}
function optnOnClick() {
    //console.log("更新成功:" + (new Date()).getTime() + "!" + e.currentTarget.id + "!");
    var req_url = "http://" + location.host + "/ajax?f=c_ques&c=CQues&m=submit_optn";
    var arrs = this.id.split("_");
    var data_obj = new Object();
    data_obj.stut_id = arrs[1];
    data_obj.excs_id = arrs[2];
    data_obj.ques_id = arrs[3];
    data_obj.optn_id = arrs[4];
    $.ajax({
        url: req_url,
        type: "POST",
        data: {
            json_str: JSON.stringify(data_obj)
        },
        dataType: "json",
        success: on_submit_optn_ok,
        error: on_submit_optn_error
    });
}
function on_submit_optn_ok(json) {
    if ('Ok' != json.status) {
        alert("提交答案失败！");
    }
}
function on_submit_optn_error(msg) {
    alert("提交答案失败：" + JSON.stringify(msg) + "！");
}
</script>