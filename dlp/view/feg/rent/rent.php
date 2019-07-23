<?php
require_once APP_ROOT.'/view/header.php';
$tabbarClassVals = array('active', '', '', '', '');
$tbIconsState = array('_sel', '', '', '', '');
$floatAreaVis = 'none';
?>
<div class="container-fluid" style="padding-left:0px;padding-right:0px;">
	<!-- header s -->
	<div id="header" class="col-xs-12 col-sm-12 col-md-12" style="margin-bottom:10px;">
	    <div class="col-xs-2 col-sm-1 col-md-1" style="padding-left:0px;padding-right:0px;text-align:left;">
	    </div>
	    <div class="col-xs-8 col-sm-10 col-md-10" style="text-align:center;">
	        <p id="pageTitle">租借提示</p>
	    </div>
	    <div class="col-xs-2 col-sm-1 col-md-1" style="padding-left:0px;padding-right:0px;text-align:right;">
	    </div>
	</div>
    <div id="rendLoadingDiv" style="width: 100%; text-align: center;">
        <img src="media/images/rent_loading.jpg" style="position: relative; width: 100%; height: 380px; top: -10px;" />
        <span style="width: 100%; font-size: 30px; color: #666666;">扫描充电宝二维码...</span>
    </div>
    <div id="confirmRentDiv" style="width: 100%; display: none;">
        <span style="width: 100%; font-size: 20px; color: #666666; text-align: center;">开始租借倒计时：</span><br /><br /><br />
        <div id="holdTime" style="width: 100%; font-size: 30px; color: #666666; text-align: center;">00:00</div><br /><br /><br />
        请确认充电宝可以正常使用。<br /><br /><br />
        <span>充电宝有问题：<a href="#">取消</a>本次租借服务</span>
    </div>
    <!-- header e -->
    <?php require_once APP_ROOT.'/view/tabbar.php'; ?>
</div>
<script type="text/javascript">
$(document).ready(function() {
    initPage();
    wx.ready(function() {
        wx.scanQRCode({
            needResult: 1, // 默认为0，扫描结果由微信处理，1则直接返回扫描结果，
            scanType: ["qrCode","barCode"], // 可以指定扫二维码还是一维码，默认>二者都有
            success: function (res) {
                var chargerCode = res.resultStr; // 当needResult 为 1 时，扫码返回的结果
                confirmRentCharger(chargerCode);
            },
            fail: function(res) {
                alert(JSON.stringify(res));
            }
        });
    });
    if (<?php echo APP_DEV?1:0; ?> == 1) {
        var chargerCode = "b01000000001";
        confirmRentCharger(chargerCode);
    }
});

var g_waitTime = 120;
function confirmRentCharger(chargerCode) {
    $("#rendLoadingDiv").css("display", "none");
    $("#confirmRentDiv").css("display", "block");
    var openId = "<?php echo $openId; ?>";
    alert("openId=" + openId + "; chargerCode=" + chargerCode + "!");
    updateHoldTime();
}

function updateHoldTime() {
    var minutes = "" + Math.floor(g_waitTime / 60);
    if (minutes.length < 2) {
        minutes = "0" + minutes;
    }
    var seconds = "" + g_waitTime % 60;
    if (seconds.length < 2) {
        seconds = "0" + seconds;
    }
    $("#holdTime").text(minutes + ":" + seconds);
    g_waitTime--;
    if (g_waitTime >= 0) {
        setTimeout(updateHoldTime, 1000);
    }
}
</script>
<?php
require_once APP_ROOT.'/view/footer.php';
?>
