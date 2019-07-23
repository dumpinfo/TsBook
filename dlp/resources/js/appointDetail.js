$(document).ready(function() {
   initPage();
   /*if(!verifyAndSetId()) {
		return;
	}*/
	/**2015-02-12**/
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
   /**2015-02-12**/
   
    var appointDoctorId;
    if($("#appointDoctorId").val() == "") {
		appointDoctorId = parseInt(localStorage.getItem("appointDoctorId"));
	} else {
		appointDoctorId = $("#appointDoctorId").val();
	}
	var hDDSPatientId;
	if($("#hDDSPatientId").val() == "") {
		hDDSPatientId = parseInt(localStorage.getItem("hDDSPatientId"));
	} else {
		hDDSPatientId = $("#hDDSPatientId").val(); 
	}
   /*获取加号详情，包括医生信息和预约信息*/
   //var appointDoctorId = parseInt(localStorage.getItem("appointDoctorId"));
   //var hDDSPatientId = parseInt(localStorage.getItem("hDDSPatientId"));
   ajaxGet("getDoctorDetail&doctorId=" + appointDoctorId , onGetDoctorDetailOk, onGetDoctorDetailError);
   ajaxGet("getAppointDetail&hDDSPatientId=" + hDDSPatientId + "&roleId=3" , onGetAppointDetailOk, onGetAppointDetailError);

   //取消加号
   $("#cancelappoint").bind("click", function() {
	   $("#cancelappointwindow").modal("show");
	   $("#cancelappointcontent").text("取消后，你将不能前往就医。确定取消吗？");
   });
   $("#confirmcancel").bind("click", function() {
	   //alert(appoint_type);
	   $("#cancelappointwindow").modal("hide");
	   if(appoint_type == 0) {
		   ajaxGet("cancelAppointByPatient&hDDSPatientId=" + hDDSPatientId, onCancelAppointOk, onCancelAppointError);
	   } else if(appoint_type == 1) {
		   var appoint_json = new Object();
		   appoint_json.hDDSPatientId = hDDSPatientId;
		   var jsonData = {
					jsonReq: JSON.stringify(appoint_json)
				  };
		    ajaxPost("patientCancelAdditionalAppoint", jsonData, onCancelAppointOk, onCancelAppointError);  
	   }
	    
   });
    //取消预约申请
   $("#cancelapply").bind("click", function() {
	  $("#cancelappointwindow").modal("show");
	  $("#cancelappointcontent").text("取消后，医生将不能看到您的预约申请。确定取消吗？");
   }); 
   
   //删除预约
   $(".deleteappoint").bind("click", function() {
	    $("#deleteappointwindow").modal("show");
        $("#deleteappointcontent").text("确定删除本条预约记录吗？");
	    $("#confirmdelete").bind("click", function() {
			   $("#deleteappointwindow").modal("hide"); 
			   ajaxGet("delAppointByPatient&hDDSPatientId=" + hDDSPatientId + "&dealType="+appoint_dealtype, onDeleteAppointOk, onDeleteAppointError);
	    });	   
   });
   //删除邀请
   $("#deleteinvitation").bind("click", function() {
	    $("#deleteappointwindow").modal("show");  
	    $("#deleteappointcontent").text("确定删除本条邀请记录吗？");
	    $("#confirmdelete").bind("click", function() {
			    $("#deleteappointwindow").modal("hide"); 
			    var appoint_json = new Object();
			    appoint_json.hDDSPatientId = hDDSPatientId;
			    var jsonData = {
						jsonReq: JSON.stringify(appoint_json)
					  };
				ajaxPost("delAdditionalAppoint", jsonData, onDeleteAppointOk, onDeleteAppointError);  
	    });	 
   }); 
   
   //重新预约
   $("#reapplybtn").bind("click", function() {
	    var doctorlistDocId = appointDoctorId;
		localStorage.setItem("doctorlistDocId", doctorlistDocId); 
        //window.location.href="doctorDetail.php";
		window.location.href=g_session.wxBaseUrl + "index.php?f=c_index&c=CIndex&m=showDoctorDetail";
   }); 
   
  
   //拒绝邀请
   $("#rejectbtn").bind("click", function() {
	    $("#rejectappointwindow").modal("show");
	    $("#confirmreject").bind("click", function() {
			    $("#rejectappointwindow").modal("hide"); 
			    var appoint_json = new Object();
				appoint_json.hDDSPatientId = hDDSPatientId;
				var jsonData = {
						jsonReq: JSON.stringify(appoint_json)
					  };
				ajaxPost("rejectAdditionalAppoint", jsonData, onRejectAppointOk, onRejectAppointError);  
	    });	
	    
   });
   
   //同意邀请
   $("#agreebtn").bind("click", function() {
	    var appoint_json = new Object();
		appoint_json.hDDSPatientId = hDDSPatientId;
		var jsonData = {
				jsonReq: JSON.stringify(appoint_json)
			  };
		ajaxPost("approveAdditionalAppoint", jsonData, onApproveAppointOk, onApproveAppointError);  
   });
   
});

function onGetDoctorDetailOk(json) {
    if(json.status == "Ok") {
	    var doc_photo = json.doctorDetail.middleAvatar;
	    var doc_name = json.doctorDetail.doctorName;
		var doc_prolevel = json.doctorDetail.professionalLevelName;
        var doc_hospital = json.doctorDetail.hospitalName;
        var doc_dept = json.doctorDetail.deptName;
		
		$("#doctorphoto").attr("src", g_session.avatarUrl + doc_photo);
		$("#doctorname").text(doc_name);
		$("#doctorprolevel").text(doc_prolevel);
		$("#doctorhospital").text(doc_hospital);
		$("#doctordept").text(doc_dept);
	} else {
	    $("#modalalert").modal("show");
	    $("#alertcontent").text("出错啦:" + json.reason + "!");
	}
} 

function onGetDoctorDetailError(e) {
    $("#modalalert").modal("show");
	$("#alertcontent").text("服务器出错：获取医生信息失败!"+e.responseText);
} 

var appoint_type;
var appoint_dealtype;
//var reply_message;
function onGetAppointDetailOk(json) {
    if(json.status == "Ok") {
	    var appoint_doctorname = json.appoint.info.doctorName;
		var appoint_time = getAppointDetailTime(json.appoint.info.startTime);
		var appoint_hospital = json.appoint.info.hospitalName;
		var appoint_dept = json.appoint.info.deptName;
		appoint_type = json.appoint.info.isAdditional;  //0是普通加号，病人发起的；1是邀请加号，医生发起的
		//reply_message = json.appoint.info.replyMessage;
		appoint_dealtype =  json.appoint.info.dealType;
		//预约加号
		//1 同意  已同意  取消加号
		//2 患者不符合要求  被拒绝  删除预约 
		//3 时间冲突  被拒绝  删除预约/重新预约
		//4 信息不全  被拒绝  删除预约/重新预约
		//5 其他原因  被拒绝  删除预约/重新预约
		//6 未通过  请等待医生审核  取消预约申请  
		//7 医生取消  医生取消  删除预约/重新预约  
		//8 患者取消  我取消  删除预约/重新预约
		
		//邀请加号
		//11 新邀请 您有一条加号邀请，请处理  拒绝/同意
		//12 同意  已同意  取消加号
		//13 拒绝  已拒绝  删除邀请
		//14 病人取消  我取消  删除邀请
		//15 医生取消  医生取消 删除邀请
		//16 拒绝  已拒绝  删除邀请
		var appoint_note;
		if(appoint_type == 0) {
			$("#appointtype").text("预约");
			$("#appointtype").css("color","red");
			
			if(appoint_dealtype == 1) {	
				$("#appointstatus").text("已通过");
				$("#actionbtn1").css("display","block");
				if(json.appoint.info.replyMessage == null) {
					appoint_note = "";
				} else {
				   appoint_note = json.appoint.info.replyMessage;
				} 
			} else if(appoint_dealtype == 2) {
				$("#appointstatus").text("被拒绝");
				$("#actionbtn2").css("display","block");
				appoint_note = "病人不符合要求";
			} else if(appoint_dealtype == 3) {
				$("#appointstatus").text("被拒绝");
				$("#actionbtn3").css("display","block");
				appoint_note = "时间冲突";
			} else if(appoint_dealtype == 4) {
				$("#appointstatus").text("被拒绝");
				$("#actionbtn3").css("display","block");
				appoint_note = "信息不全";
			} else if(appoint_dealtype == 5) {
				$("#appointstatus").text("被拒绝");
				$("#actionbtn3").css("display","block");
				if(json.appoint.info.replyMessage == null) {
					appoint_note = "";
				} else {
				   appoint_note = json.appoint.info.replyMessage;
				} 
			} else if(appoint_dealtype == 6) {
				$("#appointstatus").text("请等待医生审核");
				$("#actionbtn4").css("display","block");
				appoint_note = "";
			} else if(appoint_dealtype == 7) {
				$("#appointstatus").text("医生取消");
				$("#actionbtn3").css("display","block");
				if(json.appoint.info.replyMessage == null) {
					appoint_note = "";
				} else {
				   appoint_note = json.appoint.info.replyMessage;
				} 
			} else if(appoint_dealtype == 8) {
				$("#appointstatus").text("我取消");
				$("#actionbtn3").css("display","block");
				if(json.appoint.info.replyMessage == null) {
					appoint_note = "";
				} else {
				   appoint_note = json.appoint.info.replyMessage;
				} 
			}
			    
		} else if(appoint_type == 1) {
			
			$("#appointtype").text("邀请");
			$("#appointtype").css("color","#00CC00");
			
			if(appoint_dealtype == 11) {//新邀请
				var appoint_overDue = json.appoint.info.isOverDue;
				if(appoint_overDue == 2) {
					$("#newinvitenote").css("display","block");
					$("#statusp").hide();
					$("#actionbtn5").css("display","block");
				} else if(appoint_overDue == 1) {
					$("#appointstatus").text("已过期");
					$("#actionbtn6").css("display","block");
				}
				if(json.appoint.info.appointNote == null) {  
					appoint_note = "";
				} else {
					appoint_note = json.appoint.info.appointNote;
				}
				
			} else if(appoint_dealtype == 12) {
				$("#appointstatus").text("已通过");
				if(json.appoint.info.appointNote == null) {  
					appoint_note = "";
				} else {
					appoint_note = json.appoint.info.appointNote;
				} 
				$("#actionbtn1").css("display","block");
			} else if(appoint_dealtype == 13 || appoint_dealtype == 16) {
				$("#appointstatus").text("已拒绝");
				if(json.appoint.info.replyMessage == null) {  
					appoint_note = "";
				} else {
					appoint_note = json.appoint.info.replyMessage;
				} 
				$("#actionbtn6").css("display","block");
			} else if(appoint_dealtype == 14) {
				$("#appointstatus").text("我取消");
				if(json.appoint.info.replyMessage == null) {  
					appoint_note = "";
				} else {
					appoint_note = json.appoint.info.replyMessage;
				} 
				$("#actionbtn6").css("display","block");
			} else if(appoint_dealtype == 15) {
				$("#appointstatus").text("医生取消");
				if(json.appoint.info.replyMessage == null) {  
					appoint_note = "";
				} else {
					appoint_note = json.appoint.info.replyMessage;
				} 
				$("#actionbtn6").css("display","block");
			}	
			  
		} 
		
		$("#appointdoctorname").text(appoint_doctorname);
		$("#appointtime").text(appoint_time);
		$("#appointhospital").text(appoint_hospital);
		$("#appointdept").text(appoint_dept);
		$("#notemsg").text(appoint_note);
		
	} else {
	    $("#modalalert").modal("show");
	    $("#alertcontent").text("出错啦:" + json.reason + "!");
	}
}

function onGetAppointDetailError(e) {
    $("#modalalert").modal("show");
	$("#alertcontent").text("服务器出错：获取预约信息失败!"+e.responseText);
} 

//取消加号
function onCancelAppointOk(json) {
	if(json.status == "Ok") {
		$("#modalalert").modal("show");
	    $("#alertcontent").text("已取消!");
		$("#confirmbtn").bind("click", function() {
			if($("#isReferer").val() == 1) {
				window.history.back();	
			} else if($("#isReferer").val() == 0){
				wx.closeWindow();
			}
		});
	}
	
}
function onCancelAppointError(e) {
    $("#modalalert").modal("show");
	$("#alertcontent").text("服务器出错：取消加号失败!"+e.responseText);
}

//删除预约/邀请
function onDeleteAppointOk(json) {
	if(json.status == "Ok") {
		$("#modalalert").modal("show");
	    $("#alertcontent").text("已删除!");
		$("#confirmbtn").bind("click", function() {
			if($("#isReferer").val() == 1) {
				window.history.back();	
			} else if($("#isReferer").val() == 0){
				wx.closeWindow();
			}
		});
	}	
}
function onDeleteAppointError(e) {
    $("#modalalert").modal("show");
	$("#alertcontent").text("服务器出错：删除预约失败!"+e.responseText);
}

//拒绝邀请
function onRejectAppointOk(json) {
	if(json.status == "Ok") {
		$("#modalalert").modal("show");
	    $("#alertcontent").text("已拒绝!");
		$("#confirmbtn").bind("click", function() {
			if($("#isReferer").val() == 1) {
				window.history.back();	
			} else if($("#isReferer").val() == 0){
				wx.closeWindow();
			}
		});
	}	
}
function onRejectAppointError(e) {
    $("#modalalert").modal("show");
	$("#alertcontent").text("服务器出错：同意邀请失败!"+e.responseText);
}
//同意邀请
function onApproveAppointOk(json) {
	if(json.status == "Ok") {
		$("#modalalert").modal("show");
	    $("#alertcontent").text("已同意!");
		$("#confirmbtn").bind("click", function() {	
			if($("#isReferer").val() == 1) {
				window.history.back();	
			} else if($("#isReferer").val() == 0){
				wx.closeWindow();
			}
		});
	}	
}
function onApproveAppointError(e) {
    $("#modalalert").modal("show");
	$("#alertcontent").text("服务器出错：同意邀请失败!"+e.responseText);
}
