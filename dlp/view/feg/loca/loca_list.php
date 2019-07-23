<?php
require_once APP_ROOT.'/view/header.php';
$tabbarClassVals = array('', '', 'active', '', '');
$tbIconsState = array('', '', '_sel', '', '');
$floatAreaVis = 'block';
?>
<style type="text/css">  	 
.groupinfodiv {
    border-bottom:5px solid #e5e5e5;
}
.binded_span {
    color:#fff;
	background-color:#00CC00;
	border-radius:20px;
	padding:2px 10px;
	position:absolute;
	bottom:-78px;
	right:10px;
	z-index:0;
}
</style>

<div class="container-fluid" style="padding-left:0px;padding-right:0px;">
	<!-- header s -->
	<div id="header" class="col-xs-12 col-sm-12 col-md-12" style="margin-bottom:10px;">
	    <div class="col-xs-2 col-sm-1 col-md-1" style="padding-left:0px;padding-right:0px;text-align:left;">
		    <!--<a href="index.php?f=c_index&c=CIndex&m=showMyDoctorList" target="_parent" class="menubtn backmenu" title="返回我的医生页"></a>-->
	    </div>
	    <div class="col-xs-8 col-sm-10 col-md-10" style="text-align:center;">
	        <p id="pageTitle">合作网点列表</p>
	    </div>
	    <div class="col-xs-2 col-sm-1 col-md-1" style="padding-left:0px;padding-right:0px;text-align:right;">
	    </div>
	</div>
    <!-- header e -->
	<!-- 网点列表 -->
	<div id="locaList" class="col-xs-12 col-sm-12 col-md-12" style="margin-bottom:60px;">
	    <ul class="ullist" id="locaListUl">
        
        

		</ul>
		<div id="nodatadiv" style="display:none;text-align:center;">
			<img src="media/images/nodata.png"/>
	    </div>
    </div>
	<!-- end of locaList -->
    <?php require_once APP_ROOT.'/view/tabbar.php'; ?>
</div>

<script type="text/javascript">
$(document).ready(function() { 
    initPage();
    getLocation();
});

function getLocation() {
    if (1 == <?php echo APP_DEV?1:0; ?>) {
        var lng = 116.3;
        var lat = 40.3;
        localStorage.setItem("lng", lng);
        localStorage.setItem("lat", lat);
        ajaxGet("getLocas&lng=" + lng + "&lat=" + lat, onGetLocasOk, onGetLocasError);
        return ;
    }
    wx.error(function(res) {
    });
    wx.ready(function() {
        wx.getLocation({
            type: 'wgs84', // 默认为wgs84的gps坐标，如果要返回直接给openLocation用的火星坐标，可传入'gcj02'
            success: function (res) {
                var latitude = res.latitude; // 纬度，浮点数，范围为90 ~ -90
                var longitude = res.longitude; // 经度，浮点数，范围为180 ~ -180。
                var speed = res.speed; // 速度，以米/每秒计
                var accuracy = res.accuracy; // 位置精度
                localStorage.setItem("lng", longitude);
                localStorage.setItem("lat", latitude);
                ajaxGet("getLocas&lng=" + longitude + "&lat=" + latitude, onGetLocasOk, onGetLocasError);
            },
            fail: function(res) {
                alert("fail:" + JSON.stringify(res));
                ajaxGet("getLocas&lng=116.30&lat=40.30", onGetLocasOk, onGetLocasError);
            }
        });
    });
}

/**
* 获取合作网点列表成功
* 闫涛 2017.04.12 初始版本
*/
function onGetLocasOk(json) {
    var htmlStr = "";
    if(json.status == "Ok") {
	    var locas = json.locas;
	    if(locas.length>0) {
		    for (var i=0; i<locas.length; i++) {
                htmlStr += '<li class="row infodiv" id="item_' + locas[i].locaId + '">' + 
                            '<input type="hidden" value="' + locas[i].locaId + '" />' + 
                            '<div class="col-xs-3 col-sm-3 col-md-3 imgdiv" style="position:relative;">' + 
                                '<img class="img-circle" src="' + locas[i].locaImg + '"/>' + 
                            '</div>' + 
                            '<div class="col-xs-9 col-sm-9 col-md-9">' + 
                                '<div class="col-xs-12 col-sm-12 col-md-12" style="padding-left:5px;padding-right:5px;">' + 
                                    '<p><span class="doctorname_p">' + locas[i].locaName + '</span>&nbsp;&nbsp;<span class="appoint_info">' + locas[i].locaLevelName + '</span></p>' + 
                                    '<p class="ifoverflow appoint_info">安卓：' + locas[i].androidNums + '支；苹果：' + locas[i].iosNums + '支</p>' + 
                                    '<p class="ifoverflow appoint_info">距离：' + locas[i].distance + '内</p>' + 
                                '</div>' + 
                            '</div>' + 
                            '</li>';
			}
            $("#locaListUl").html(htmlStr);
            // 绑定条目单击事件
		} else {
		    $("#locaListUl").hide();
		    $("#nodatadiv").css({"display":"block","margin-top": wheight/2 - 100});
		}
	    
    } else {
	    if(g_session.debug == true) {
			$("#modalalert").modal("show");
			$("#alertcontent").text("获取共享便利店网点列表失败：" + json.reason);
		}
	}
} 
/**
* 获取合作网点列表失败
*【闫涛 2017.04.13】初始版本
*/
function onGetLocasError(e) {
    $("#modalalert").modal("show");
    if(g_session.debug == true) {	
		$("#alertcontent").text("服务器出错：获取我的医生列表失败！"+e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
}
</script>
<?php
require_once APP_ROOT.'/view/footer.php';
?>
