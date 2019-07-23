var g_session = {
        "debug":false,
	    "dwkBaseUrl":"http://"+document.location.host+"/a/limesurvey/dwk/index.php?",
	    "serverBaseUrl":"http://"+document.location.host+"/a/index.php?",
		"avatarUrl":"http://"+document.location.host+"/a/",
		"sftBaseUrl":"http://"+document.location.host+"/a/",
		"wxBaseUrl":"http://"+document.location.host+"/a/wechat/",
	};
	
//提示框，一个确定按钮
var alertString = '<div class="modal" id="modalalert">'+
  '<div class="modal-dialog modal-sm" style="width:300px;">'+
    '<div class="modal-content">'+
      '<div class="modal-body">'+
        '<p style="font-size:16px;" id="alertcontent">提醒</p>'+
      '</div>'+
      '<div class="modal-footer">'+
        '<button type="button" class="btn btn-success" data-dismiss="modal" id="confirmbtn">确定</button>'+
      '</div>'+
    '</div>'+
  '</div>'+
'</div>';

var wheight = $(window).height();
var wwidth = $(window).width();	
var content_height;
function initPage() {  	
    $(".blackbox").css("top",(document.body.scrollTop?document.body.scrollTop:document.documentElement.scrollTop)+wheight*2/3+"px"); 
    $(".blackbox").css({"width":wwidth/4+"px","height":wwidth/4+"px"})
	$(".blackbox").css({"left":wwidth/2-wwidth/8+"px"});
    
	content_height = $(".customcontent").height();
	$("#alertdiv").css({"padding-top":(wwidth/4-content_height)/2+"px"});
	
    $("#loading img").css({"padding-top":wwidth/8-20+"px"});
	
    $("#loading").bind("ajaxSend", function() {
        $("#loading").css("display","block");  	
    });

	$("#loading").bind("ajaxComplete", function() {
		$("#loading").css("display","none");  	
	});
	
	$("body").append(alertString);
	
    $(".modal-dialog").css("margin-top", (wheight-150)*2/3 + "px");
    $(".modal-dialog").css("margin-left", (wwidth-300)/2 + "px");
	
	$("#modalwindow .modal-dialog").css("margin-top",(wheight-150)*2/3 + "px");
}

var openId;
var doctorId;
var userId;
var organizationId;
function verifyAndSetId() {	
	if($("#doctorId").val() == "" && localStorage.getItem("doctorId") == null && $("#organizationId").val() == "" && localStorage.getItem("organizationId") == null && $("#userId").val() == "" && localStorage.getItem("userId") == null) {
	    $("#modalalert").modal("show");
	    $("#alertcontent").text("您只有按照微信提示完善个人信息后才可以使用!");
		$("#confirmbtn").bind("click", function() {
			wx.closeWindow();
		});	
		return false;
	}
	if(localStorage.getItem("userId") == 0 && $("#userId").val() != 0) {
               userId = $("#userId").val();
               localStorage.setItem("userId", userId);
    } else if(localStorage.getItem("userId") == 0 || $("#userId").val() === 0) {
		$("#modalalert").modal("show");
		$("#alertcontent").text("您只有按照微信提示完善个人信息后才可以使用!");
		$("#confirmbtn").bind("click", function() {
			wx.closeWindow();
		});			 
		return false;
	}
	
	if($("#openId").val() == "") {
        openId = localStorage.getItem("openId");
	} else {
		openId = $("#openId").val();
		localStorage.removeItem("openId");
		localStorage.setItem("openId", openId); 
	}
	if($("#doctorId").val() == "") {
		doctorId = parseInt(localStorage.getItem("doctorId"));
	} else {
		doctorId = $("#doctorId").val();
		localStorage.setItem("doctorId", doctorId); 
	}
	if($("#userId").val() == "") {
		userId = parseInt(localStorage.getItem("userId"));
	} else {
		userId = $("#userId").val();
		localStorage.setItem("userId", userId);  
	}
	
	return true;
}

var doctorName;
function verifyAndSetIdForRegister() {	
	if($("#openId").val() == "") {
        openId = localStorage.getItem("openId");
	} else {
		openId = $("#openId").val();
		localStorage.removeItem("openId");
		localStorage.setItem("openId", openId); 
	}
	if($("#doctorId").val() == "") {
		doctorId = parseInt(localStorage.getItem("doctorId"));
	} else {
		doctorId = $("#doctorId").val();
		localStorage.setItem("doctorId", doctorId); 
	}
	if($("#organizationId").val() == "") {
		organizationId = parseInt(localStorage.getItem("organizationId"));
	} else {
		organizationId = $("#organizationId").val();
		localStorage.setItem("organizationId", organizationId); 
	}
    if($("#userId").val() == "") {
		userId = parseInt(localStorage.getItem("userId"));
	} else {
		userId = $("#userId").val();
		localStorage.setItem("userId", userId);  
	} 
	if($("#doctorName").val() == "") {
		doctorName = localStorage.getItem("doctorName");
	} else {
		doctorName = $("#doctorName").val();
		localStorage.setItem("doctorName", doctorName);  
	} 
	
}

function ajaxGet(interfaceName, onSuccess, onError) {
    $.ajax({
        url: 'index.php?f=c_ajax&c=CAjax&m=' + interfaceName,
        type: 'GET',
        dataType: 'json',
        success: onSuccess,
        error: onError
    });
}

function ajaxPost(interfaceName, jsonData, onSuccess, onError) {
    $.ajax({
		url: 'index.php?f=c_ajax&c=CAjax&m=' + interfaceName,
		type: 'POST',
		data: jsonData,
		dataType: 'json',
		success: onSuccess,
		error: onError
	  });
}

function ajaxGet_sync(interfaceName, onSuccess, onError) {
    $.ajax({
        url: 'index.php?f=c_ajax&c=CAjax&m=' + interfaceName,
        type: 'GET',
		async: false,
        dataType: 'json',
        success: onSuccess,
        error: onError
    });
}

//日期格式为 2013-03-10 00:00:00,返回2012/03/10
function getDateCustom(dateString) {
    if(!isEmpty(dateString)) {
	  var dateString = dateString.substr(0,10);
	  var dateArray = dateString.split('-');
		
	  var ssdate= dateArray[0]+"/"+dateArray[1]+"/"+dateArray[2];
	  return ssdate;
    }
}

//日期格式为 2013-03-10 00:00:00,返回周
function getWeekDayCustom(dateString) {
    if(!isEmpty(dateString)) {
		var dateString = dateString.substr(0,10);    
		var dateArray = dateString.split('-');
		
		var ssdate=new Date(dateArray[0], parseInt(dateArray[1]-1), dateArray[2]);

		var weekArray = new Array("周日", "周一", "周二", "周三", "周四", "周五", "周六");	
		var weekDay = weekArray[ssdate.getDay()];
		return weekDay;
	}
}

//日期格式为 2013-03-10 00:00:00,返回上午，下午或晚上
function getTimeCustom(dateString) {
    if(!isEmpty(dateString)) {
		var timeString = dateString.substr(11,8);      
		
		var timeSeg = "";
		if(timeString >= "06:00:00" && timeString <= "12:30:00") {
			timeSeg = "上午";
		} else if(timeString >= "13:00:00" && timeString <= "18:30:00") {
			timeSeg = "下午";
		} else if(timeString >= "19:00:00" && timeString <= "23:30:00") {
			timeSeg = "晚上";
		}
		return timeSeg;
	}
}

//日期格式为 2013-03-10 19:00:00,返回时间19:00
function getTime(dateString) {
    if(!isEmpty(dateString)) {
		var timeString = dateString.substr(11,5);      	
		return timeString;
	}
}

//日期格式为 2013-03-10 00:00:00,返回2015/01/01,星期四下午
function getAppointTime(dateString) {
    if(!isEmpty(dateString)) {
		var appoint_date = getDateCustom(dateString);
		var appoint_week = getWeekDayCustom(dateString);
		var appoint_time = getTimeCustom(dateString);
		
		var appointTime = appoint_date + "，" + appoint_week + appoint_time; 
		return appointTime;
	}
}

//日期格式为 2013-03-10 00:00:00,返回2015-01-01,星期四下午
function getAppointDetailTime(dateString) {
    if(!isEmpty(dateString)) {
		var appoint_date = dateString.substr(0,10);
		var appoint_week = getWeekDayCustom(dateString);
		var appoint_time = getTimeCustom(dateString);
		
		var appointTime = appoint_date + "，" + appoint_week + appoint_time; 
		return appointTime;
	}
}

//获取某个指定日期时间的时间戳
function getUnixTimestamp(endTime){ 
	var date=new Date(); 
	date.setFullYear(endTime.substring(0,4)); 
	date.setMonth(endTime.substring(5,7)-1); 
	date.setDate(endTime.substring(8,10)); 
	date.setHours(endTime.substring(11,13)); 
	date.setMinutes(endTime.substring(14,16)); 
	date.setSeconds(endTime.substring(17,19)); 
	return Date.parse(date)/1000; 
	}
//判断两个日期相差天数
//startDate和endDate都是标准日期格式 Thu Jul 30 2015 12:L47:51 GMT+0800
function GetDateDiff(startDate,endDate) 
{ 
    //var startTime = new Date(Date.parse(startDate.replace(/-/g, "/"))).getTime(); 
    //var endTime = new Date(Date.parse(endDate.replace(/-/g, "/"))).getTime(); 
    var dates = (endDate - startDate)/(60*60*24); 
    return dates; 
} 

//按照某一属性降序排列
function compare(propertyName) { 
    return function (object1, object2) { 
        var value1 = object1[propertyName]; 
        var value2 = object2[propertyName]; 
        if (value2 < value1) { 
            return -1; 
        }else if (value2 > value1) { 
            return 1; 
        } else { 
            return 0; 
        } 
    } 
}

//获取URL字符串后的参数
function getUrlParam(url,name){  
     var pattern = new RegExp("[?&]"+name+"\=([^&]+)", "g");  
     var matcher = pattern.exec(url);  
     var items = null;  
     if(null != matcher){  
             try{  
                    items = decodeURIComponent(decodeURIComponent(matcher[1]));  
             }catch(e){  
                     try{  
                             items = decodeURIComponent(matcher[1]);  
                     }catch(e){  
                             items = matcher[1];  
                     }  
             }  
     }  
     return items;  
}

//获取字符串的长度，英文是一个长度，中文是两个长度
function strLength(str) {
	var len = 0;
	for (var i = 0; i < str.length; i++) {
		var c = str.charCodeAt(i);
		//单字节加1 
		if ((c >= 0x0001 && c <= 0x007e) || (0xff60 <= c && c <= 0xff9f)) {
			len++;
		}
		else {
			len += 2;
		}
	}
	return len;
}

function isEmpty(value) {
    if(value == null || value == undefined || value.trim() == "") {
	    return true;
	}
}

/*设置单选框样式*/
function radio_style() {//3设置样式 
  if ($("input")) { 
    var r = $("input"); 

    function select_element(obj, type) {//1设置背景图片 type的值是1时，表示未选中 
      $(obj).parent().css({"background":"url("+g_session.wxBaseUrl+"/media/images/circle01.png) no-repeat","background-size":"15px 15px","background-position":"0px 5px"});
      if (type) {
        $(obj).parent().css({"background":"url("+g_session.wxBaseUrl+"/media/images/circle010.png) no-repeat","background-size":"15px 15px","background-position":"0px 5px"});
      } 
    }//1 

    for (var i = 0; i < r.length; i++) { 
      if (r[i].type == "radio") { 
        r[i].style.opacity = 0; 
        r[i].style.filter = "alpha(opacity=0)";//隐藏真实的radio 

        r[i].onclick = function() { 
			select_element(this); 
			unfocus(); 
		} 
        if (r[i].checked == true) {
	      select_element(r[i]);
	    } else { 
	    select_element(r[i], 1); 
	    } 
      } 
    } 

    function unfocus() {//2处理未选中 
      for (var i = 0; i < r.length; i++) { 
        if (r[i].type == "radio") { if (r[i].checked == false) { select_element(r[i], 1) } } 
      } 
    } //2 
  } 
} //3
  
String.prototype.trim=function(){
	return this.replace(/(^\s*)|(\s*$)/g, "");
}
String.prototype.ltrim=function(){
	return this.replace(/(^\s*)/g,"");
}
String.prototype.rtrim=function(){
	return this.replace(/(\s*$)/g,"");
}
/**
*
*  Base64 encode / decode
*
*
*/
function Base64() {
 
	// private property
	_keyStr = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
 
	// public method for encoding
	this.encode = function (input) {
		var output = "";
		var chr1, chr2, chr3, enc1, enc2, enc3, enc4;
		var i = 0;
		input = _utf8_encode(input);
		while (i < input.length) {
			chr1 = input.charCodeAt(i++);
			chr2 = input.charCodeAt(i++);
			chr3 = input.charCodeAt(i++);
			enc1 = chr1 >> 2;
			enc2 = ((chr1 & 3) << 4) | (chr2 >> 4);
			enc3 = ((chr2 & 15) << 2) | (chr3 >> 6);
			enc4 = chr3 & 63;
			if (isNaN(chr2)) {
				enc3 = enc4 = 64;
			} else if (isNaN(chr3)) {
				enc4 = 64;
			}
			output = output +
			_keyStr.charAt(enc1) + _keyStr.charAt(enc2) +
			_keyStr.charAt(enc3) + _keyStr.charAt(enc4);
		}
		return output;
	}
 
	// public method for decoding
	this.decode = function (input) {
		var output = "";
		var chr1, chr2, chr3;
		var enc1, enc2, enc3, enc4;
		var i = 0;
		input = input.replace(/[^A-Za-z0-9\+\/\=]/g, "");
		while (i < input.length) {
			enc1 = _keyStr.indexOf(input.charAt(i++));
			enc2 = _keyStr.indexOf(input.charAt(i++));
			enc3 = _keyStr.indexOf(input.charAt(i++));
			enc4 = _keyStr.indexOf(input.charAt(i++));
			chr1 = (enc1 << 2) | (enc2 >> 4);
			chr2 = ((enc2 & 15) << 4) | (enc3 >> 2);
			chr3 = ((enc3 & 3) << 6) | enc4;
			output = output + String.fromCharCode(chr1);
			if (enc3 != 64) {
				output = output + String.fromCharCode(chr2);
			}
			if (enc4 != 64) {
				output = output + String.fromCharCode(chr3);
			}
		}
		output = _utf8_decode(output);
		return output;
	}
 
	// private method for UTF-8 encoding
	_utf8_encode = function (string) {
		string = string.replace(/\r\n/g,"\n");
		var utftext = "";
		for (var n = 0; n < string.length; n++) {
			var c = string.charCodeAt(n);
			if (c < 128) {
				utftext += String.fromCharCode(c);
			} else if((c > 127) && (c < 2048)) {
				utftext += String.fromCharCode((c >> 6) | 192);
				utftext += String.fromCharCode((c & 63) | 128);
			} else {
				utftext += String.fromCharCode((c >> 12) | 224);
				utftext += String.fromCharCode(((c >> 6) & 63) | 128);
				utftext += String.fromCharCode((c & 63) | 128);
			}
 
		}
		return utftext;
	}
 
	// private method for UTF-8 decoding
	_utf8_decode = function (utftext) {
		var string = "";
		var i = 0;
		var c = c1 = c2 = 0;
		while ( i < utftext.length ) {
			c = utftext.charCodeAt(i);
			if (c < 128) {
				string += String.fromCharCode(c);
				i++;
			} else if((c > 191) && (c < 224)) {
				c2 = utftext.charCodeAt(i+1);
				string += String.fromCharCode(((c & 31) << 6) | (c2 & 63));
				i += 2;
			} else {
				c2 = utftext.charCodeAt(i+1);
				c3 = utftext.charCodeAt(i+2);
				string += String.fromCharCode(((c & 15) << 12) | ((c2 & 63) << 6) | (c3 & 63));
				i += 3;
			}
		}
		return string;
	}
}
