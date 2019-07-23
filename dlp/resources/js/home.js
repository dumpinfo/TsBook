$(document).ready(function() {
    initPage();
	verifyAndSetIdForRegister();
    $(".yourdoctorname").text(doctorName);
	radio_style();
	getTextMessage();
	
	//111111
	/*if(parseInt(localStorage.getItem("register_isClicked")) == 1) {
		var init_phone = localStorage.getItem("rphone");
		if(init_phone && init_phone!="") {
			$("#rphone").val(init_phone);
		} else {
			$("#rphone").val("");
		}
		
		var init_userType = localStorage.getItem("userType");
		if(init_userType && init_userType!="") {
			userType = init_userType;
		} 
		
	    $("#register").hide();
		$("#yourphoneno").text(init_phone);
	    $("#2next").show();	
        timedCount();		
	}*/
	
    $("#vcodelink").bind("click", function() { 
	   var phonenumber = $("#rphone").val();
	   var re = /^1\d{10}$/;
	   
      if(phonenumber == ""||(re.test(phonenumber) == false)) {
		 $("#modalalert").modal("show");
		 $("#alertcontent").text("请填写11位的手机号码！");
	   } else {	
	       $("#vcodelink").addClass("disablelink");
		   $("#vcodelink").attr("disabled","true");
		   
	       ajaxGet("getUserType&loginName=" + phonenumber + "&roleId=3", onGetUserTypeOk, onGetUserTypeError);	   
	   }
	});
	
	//alert(userType);
	$("#abutton").bind("click", function() {
	    var phonenumber = $("#rphone").val();
		if(userType == 1) {
			getVcode(phonenumber,2,1);
		} else {
			getVcode(phonenumber,2,2);
		}

	});
	
	
	/*****以下涉及注册流程第三步****/
	//radio_style();
	/*获取用户个人资料*/
    ajaxGet("getWeiXinUserInfo&openId=" + openId, onGetUserInfoOk, onGetUserInfoError);
	
	$(".plugindate").hide();
	//解决IE等浏览器不支持html5 input type=date
	var initDatepicker = function() { 
      //datetimepicker
      $('.plugindate').datetimepicker({
        language:  'zh-CN',
	    format: "yyyy-mm-dd",
        weekStart: 1,
        todayBtn:  1,
		autoclose: 1,
		todayHighlight: 1,
		startView: 2,
		minView: 2,
		forceParse: 0
      });
    }; 
	
    if(!Modernizr.inputtypes.date){ 
	  $(".html5date").hide();
      $(".plugindate").show();	
      $(document).ready(initDatepicker); 
	  
      $(".plugindate").bind("blur", function() {
		 birthdate = $(this).val();
	  });	  
    }; 
	
	$(".html5date").bind("blur", function() {
		birthdate = $(this).val();
	});

});

var birthdate = "";
var token = "";
function getTextMessage() {
    $.ajax({
        url: g_session.serverBaseUrl + 'f=c_ajax&c=CAjax&m=getTextMessage',
        type: 'GET',
        dataType: 'json',
        success: function(json) {
		    token = json.token;
		},
        error: function(e) {
		    $("#modalalert").modal("show");
			if(g_session.debug == true) {	
				$("#alertcontent").text("获取信息失败！"+e.responseText);
			} else {
				$("#alertcontent").text("服务器异常，请稍后重试！");
			}
		}
    });
}

function onGetUserInfoOk(json) {
    if(json.status == "Ok") {
		var user_name = json.weixinInfo.nickname;
		var user_gender = json.weixinInfo.sex;

		$("#rname").val(user_name);
		if(user_gender == 1) {
		    
            $('input[name="rgender"][value="0"]').attr("checked",true);			
		} else if(user_gender == 2) {
			$('input[name="rgender"][value="1"]').attr("checked",true);
		}
		radio_style();
		
	} else {
		$("#modalalert").modal("show");
		$("#alertcontent").text(json.reason);
	}
}
function onGetUserInfoError(e) {
    $("#modalalert").modal("show");
    if(g_session.debug == true) {	
		$("#alertcontent").text("服务器出错：获取用户个人资料失败！"+e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
}

var userType;
function onGetUserTypeOk(json) {
    if(json.status == "Ok") {
	    userType = json.type;
		//222222
		localStorage.setItem("userType", userType);
		
		var phonenumber = $("#rphone").val();  
		timedCount();
		
		$("#loading").css("display","block");  
		if(userType == 1) {
		    getVcode(phonenumber,2,1);
		} else {
		    getVcode(phonenumber,2,2);
		}		  
	} else {
	    $("#vcodelink").removeClass("disablelink");
		$("#vcodelink").removeAttr("disabled");

		$("#modalalert").modal("show");
		$("#alertcontent").text(json.reason);
	}
}

function onGetUserTypeError(e) {
    $("#vcodelink").removeClass("disablelink");
	$("#vcodelink").removeAttr("disabled");	
	$("#modalalert").modal("show");
	if(g_session.debug == true) {	
		$("#alertcontent").text("服务器出错：获取用户类型失败！"+e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
}

function getVcode(phonenumber,type,userType) { 
	if(type == 1) {//type 1是短信验证码，2是语音验证码   
       //verifyType=userType 1是注册 2是找回密码
       //ajaxGet("getFlag&verifyType="+userType+"&phoneNum=" + phonenumber + "&roleId=3", onGetSmsVcodeOk, onGetSmsVcodeError);
	    var rawReqUrl = g_session.serverBaseUrl + 'cmd=getFlag&verifyType='+userType+'&phoneNum=' + phonenumber + '&roleId=3';
	    var pUrl = 'index.php?cmd=getFlag&verifyType='+userType+'&phoneNum=' + phonenumber + '&roleId=3' + token;
	    var reqUrl = rawReqUrl + "&mac=" + hex_sha1(pUrl);
	    $.ajax({
			url: reqUrl,
			type: 'GET',
			dataType: 'json',
			success: onGetSmsVcodeOk,
			error: onGetSmsVcodeError
        });
		
	} else if(type == 2) {
	
		$("#abutton").addClass("disablelink");
		$("#abutton").attr("disabled","true");
	    timedCount1();
		setTimeout(function() {
			   $("#abutton").removeClass("disablelink");
			   $("#abutton").removeAttr("disabled");
		 },60000);
	   //ajaxGet("getVoiceFlag&verifyType="+userType+"&phoneNum=" + phonenumber + "&roleId=3", onGetVoiceVcodeOk, onGetVoiceVcodeError);
	   
	    
	    var rawReqUrl = g_session.serverBaseUrl + 'cmd=getVoiceFlag&verifyType='+userType+'&phoneNum=' + phonenumber + '&roleId=3';
		var pUrl = 'index.php?cmd=getVoiceFlag&verifyType='+userType+'&phoneNum=' + phonenumber + '&roleId=3' + token;
	    var reqUrl = rawReqUrl + "&mac=" + hex_sha1(pUrl);
	    $.ajax({
			url: reqUrl,
			type: 'GET',
			dataType: 'json',
			success: onGetVoiceVcodeOk,
			error: onGetVoiceVcodeError
        });
	}	
}

//var vCode;
//var phoneNumber;
function onGetSmsVcodeOk(json) {
    $("#vcodelink").removeClass("disablelink");
	$("#vcodelink").removeAttr("disabled");
	
	if(json.status=="Ok") {
		//vCode = json.verifyCode;
		//phoneNumber = json.phoneNum;
		
		//222222
		var phonenumber = $("#rphone").val();
		localStorage.setItem("rphone", phonenumber);
		localStorage.setItem("register_isClicked", 1);
		  
		$("#register").hide();
		$("#yourphoneno").text(json.phoneNum);
	    $("#2next").show();	

	} else {

		 clearTimeout(timer);
		 $("#count").val("");
		 $("#count").hide();
		 $("#abutton").show();
		 
		 $("#modalalert").modal("show");
		 $("#alertcontent").text(json.reason); 
		
		 return;
	}
	
}
function onGetSmsVcodeError(e) {
    $("#vcodelink").removeClass("disablelink");
	$("#vcodelink").removeAttr("disabled");
	
	clearTimeout(timer);
	$("#count").val("");
	$("#count").hide();
	$("#abutton").show();
	
    $("#modalalert").modal("show");
    if(g_session.debug == true) {		
		$("#alertcontent").text("服务器出错：获取短信验证码失败！" +e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
	return;
}

var c = 60; 
var timer = null;
function timedCount() {
  if(c == 0) {
	  $("#count").val("");
	  $("#count").hide();
	  $("#abutton").show();
	  c = 60;
	  return false;
  } else {
	   $("#count").val(c + "秒后可获取语音验证码");
       c = c-1;
       timer = setTimeout("timedCount()",1000);
  }
}

//var voice_vCode;
function onGetVoiceVcodeOk(json) {
    $("#vcodelink").removeClass("disablelink");
	$("#vcodelink").removeAttr("disabled");
	
	if(json.status=="Ok") {
		 //voice_vCode = json.verifyCode;
		 //phoneNumber = json.phoneNum;
		 
		//222222
		var phonenumber = $("#rphone").val();
		localStorage.setItem("rphone", phonenumber);
		localStorage.setItem("register_isClicked", 1);
		  
		$("#register").hide();
		$("#yourphoneno").text(json.phoneNum);
	    $("#2next").show();	
	} else {
		 clearTimeout(timer1);
		 $("#abutton").removeClass("disablelink");
		 $("#abutton").removeAttr("disabled");
		 $("#abutton").val("获取语音验证码");
		 
		 $("#modalalert").modal("show");
		 $("#alertcontent").text("获取语音验证码失败，请60秒后重试！"); 
	}
	
}
function onGetVoiceVcodeError(e) {
	clearTimeout(timer1);	
	$("#abutton").removeClass("disablelink");
	$("#abutton").removeAttr("disabled");
	$("#abutton").val("获取语音验证码");
	
	$("#modalalert").modal("show");
    if(g_session.debug == true) {		
		$("#alertcontent").text("服务器出错：获取语音验证码失败！" +e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
	
}

var c1 = 60; 
var timer1 = null;
function timedCount1() {
  if(c1 == 0) {
	  $("#abutton").removeClass("disablelink");
	  $("#abutton").removeAttr("disabled");
	  $("#abutton").val("获取语音验证码");
	  c1 = 60;
	  return false;
  } else {
	   $("#abutton").val(c1 + "秒后重新获取语音验证码");
       c1 = c1-1;
       timer1 = setTimeout("timedCount1()",1000);
  }
}

function submitRegisterForm() {	
    var verify_code = $("#vcode").val();	
	if(verify_code != "") {
	    $("#submitbutton").attr("disabled","true");
	    $("#submitbutton").addClass("disabledbtn");	
		
		var phonenumber = $("#rphone").val();
		ajaxGet("isPassVerify&verifyCode="+verify_code+"&phoneNum=" + phonenumber + "&roleId=3", onVerifyVcodeOk, onVerifyVcodeError);
		
		//333333
        removeLocalstorageItem();
	} else {
	    $("#modalalert").modal("show");
		$("#alertcontent").text("验证码不能为空！");
	}
	
}

//userType 1代表未注册 2代表已注册未激活 3代表已注册已激活
//1流程：填写手机号，获取验证码getFlag，验证验证码有效性isPassVerify，填写姓名，绑定用户到微信bindUserToWeixin
//2流程：填写手机号，获取验证码getFlag，验证验证码有效性isPassVerify，获取激活码getActiveCode，激活activeUser，绑定用户到微信bindUserToWeixin
//3流程：填写手机号，获取验证码getFlag，验证验证码有效性isPassVerify，绑定用户到微信bindUserToWeixin
function onVerifyVcodeOk(json) {
    if(json.status == "Ok") {				
			if(userType == 1) {
				$("#submitbutton").removeAttr("disabled");
				$("#submitbutton").removeClass("disabledbtn");
				
				$("#2next").hide();
				$("#registername").show();
				
				$("#registernamebtn").bind("click", function() {
					var userName = $("#rname").val();
					var loginName = $("#rphone").val();
					var userGender = $("input[name='rgender']:checked").val();
					var userBirthday = birthdate;
					
					if(userName != "" && strLength(userName)<=12) {
						var jsonreq = new Object();
						jsonreq.openId = openId;
						
						if($("#doctorId").val() != "") {
							jsonreq.doctorId = doctorId;
						} else if($("#organizationId").val() != "") {
							jsonreq.organizationId = organizationId;
						}
						jsonreq.isNew = 1;
						jsonreq.userName = userName;
						jsonreq.loginName = loginName;
						jsonreq.gender = userGender;
						jsonreq.birthday = userBirthday;
						
						var jsonData = {
							jsonReq: JSON.stringify(jsonreq)
						  };
						  
						$("#registernamebtn").attr("disabled","true");
						$("#registernamebtn").addClass("disabledbtn");
						
						ajaxPost("bindUserToWeixin", jsonData, onRegisterOk, onRegisterError); 
					} else {
						$("#modalalert").modal("show");
						$("#alertcontent").text("姓名不能为空！");
					}
					
				});
			   
			} else if(userType == 2) {
				var loginName = $("#rphone").val();
				ajaxGet("getActiveCode&loginName=" + loginName + "&roleId=3", onGetAcodeOk, onGetAcodeError);
			  
			} else if(userType == 3) {
				var loginName = $("#rphone").val();
				var loginPwd = loginName.split("").reverse().join(""); 
			  
				var jsonreq = new Object();
				jsonreq.openId = openId;
				
				if($("#doctorId").val() != "") {
					jsonreq.doctorId = doctorId;
				} else if($("#organizationId").val() != "") {
					jsonreq.organizationId = organizationId;
				}
				jsonreq.isNew = 0;
				jsonreq.loginName = loginName;
				jsonreq.loginPwd = hex_md5(loginPwd);
			  
				var jsonData = {
					jsonReq: JSON.stringify(jsonreq)
				  };
				ajaxPost("bindUserToWeixin", jsonData, onLoginOk, onLoginError); 
			}	
	   
	} else {
		$("#modalalert").modal("show");
		$("#alertcontent").text("请填写正确的验证码！");

		$("#submitbutton").removeAttr("disabled");
	    $("#submitbutton").removeClass("disabledbtn");
	}
}

function onVerifyVcodeError(e) {	
	$("#modalalert").modal("show");
	if(g_session.debug == true) {		
		$("#alertcontent").text("服务器出错：检测验证码是否正确失败！" +e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
	$("#submitbutton").removeAttr("disabled");
	$("#submitbutton").removeClass("disabledbtn");
}

/**获取激活码**/
function onGetAcodeOk(json) {	
    if(json.status == "Ok") {
		var aCode = json.activeCode;
		var loginName = $("#rphone").val();
		var loginPwd = loginName.split("").reverse().join(""); 
		
		var jsonreq = new Object();
		jsonreq.mobilePhone = loginName;
		jsonreq.activeCode = aCode;
		jsonreq.password = hex_md5(loginPwd);
		jsonreq.roleId = 3;
		
		var jsonData = {
			jsonReq: JSON.stringify(jsonreq)
		  };
		ajaxPost("activeUser", jsonData, onActiveOk, onActiveError);
	} else {
        $("#submitbutton").removeAttr("disabled");
	    $("#submitbutton").removeClass("disabledbtn");
		
		$("#modalalert").modal("show");
		$("#alertcontent").text(json.reason); 
	}
}

function onGetAcodeError(e){
    $("#submitbutton").removeAttr("disabled");
	$("#submitbutton").removeClass("disabledbtn");
	
	$("#modalalert").modal("show");
    if(g_session.debug == true) {		
		$("#alertcontent").text("服务器出错：获取激活码失败！"+e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
}

/**注册**/
function onRegisterOk(json) {
    $("#registernamebtn").removeClass("disablelink");
	$("#registernamebtn").removeAttr("disabled");
	
	//激活后bind
	$("#submitbutton").removeAttr("disabled");
	$("#submitbutton").removeClass("disabledbtn");
	
    if(json.status == "Ok") {		
		$("#alertdiv").fadeIn(400);
		$(".customcontent").html("登录<br/>成功");
		content_height = $(".customcontent").height();
		$("#alertdiv").css({"padding-top":(wwidth/4-content_height)/2+"px"});
					
		setTimeout(function() {
				$("#alertdiv").fadeOut(1000);
				wx.closeWindow();
			},2000);				
	} else {		
		$("#modalalert").modal("show");
		$("#alertcontent").text(json.reason); 
	}
}

function onRegisterError(e) {	
    $("#registernamebtn").removeClass("disablelink");
	$("#registernamebtn").removeAttr("disabled");
	
    //激活后bind
	$("#submitbutton").removeAttr("disabled");
	$("#submitbutton").removeClass("disabledbtn");
	
	$("#modalalert").modal("show");
	if(g_session.debug == true) {
		$("#alertcontent").text("服务器出错：注册失败！"+e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
	 
}

/**激活**/
function onActiveOk(json) {
    if(json.status == "Ok") {
	    var loginName = $("#rphone").val();
	    var loginPwd = loginName.split("").reverse().join(""); 
			
	    var jsonreq = new Object();
	    jsonreq.openId = openId;
		
	    if($("#doctorId").val() != "") {
			jsonreq.doctorId = doctorId;
		} else if($("#organizationId").val() != "") {
			jsonreq.organizationId = organizationId;
		}
	    jsonreq.isNew = 0;
	    jsonreq.loginName =  loginName;
	    jsonreq.loginPwd = hex_md5(loginPwd);
	    jsonreq.isActive = 1;
		
		var jsonData = {
			jsonReq: JSON.stringify(jsonreq)
		};
		ajaxPost("bindUserToWeixin", jsonData, onRegisterOk, onRegisterError); 
	} else {
	    $("#submitbutton").removeAttr("disabled");
	    $("#submitbutton").removeClass("disabledbtn");	
		
		$("#modalalert").modal("show");
		$("#alertcontent").text(json.reason); 
	}

}

function onActiveError(e) {
    $("#submitbutton").removeAttr("disabled");
	$("#submitbutton").removeClass("disabledbtn");

    $("#modalalert").modal("show");	
	if(g_session.debug == true) {	
		$("#alertcontent").text("服务器出错：激活失败！"+e.responseText);
	} else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
}

/**登录**/
function onLoginOk(json) {		
    $("#submitbutton").removeAttr("disabled");
	$("#submitbutton").removeClass("disabledbtn");
	
    if(json.status == "Ok") {		
		$("#alertdiv").fadeIn(400);
		$(".customcontent").html("登录<br/>成功");
		content_height = $(".customcontent").height();
		$("#alertdiv").css({"padding-top":(wwidth/4-content_height)/2+"px"});
		
		setTimeout(function() {
				$("#alertdiv").fadeOut(1000);
				wx.closeWindow();
			},2000);		
	} else {		
		$("#modalalert").modal("show");
		$("#alertcontent").text(json.reason);		
	}
}

function onLoginError(e) {	
    $("#submitbutton").removeAttr("disabled");
	$("#submitbutton").removeClass("disabledbtn");
	$("#modalalert").modal("show");
	if(g_session.debug == true) {	
		$("#alertcontent").text("服务器出错：登录失败！"+e.responseText);
	}else {
	    $("#alertcontent").text("服务器异常，请稍后重试！");
	}
}

function removeLocalstorageItem() {
    localStorage.removeItem("register_isClicked");
	localStorage.removeItem("rphone");
}