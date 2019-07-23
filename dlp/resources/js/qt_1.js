
<script type="text/javascript">
$("div[id^='o_'").each(bind_option_click);

function bind_option_click(idx, elem) {
    $("#" + elem.id).bind("click", optnOnClick);
}
function optnOnClick(e) {
    //console.log("更新成功:" + (new Date()).getTime() + "!" + e.currentTarget.id + "!");
}
</script>